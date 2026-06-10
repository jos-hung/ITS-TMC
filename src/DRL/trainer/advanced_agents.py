import random
from collections import deque
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from configs.systemcfg import DEVICE, GLOBAL_SEED, a2c_cfg, ddpg_cfg, ppo_cfg, eval as eval_mode


if DEVICE != "cpu":
    device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")


STATE_CLIP = 1e4
REWARD_CLIP = 1e6
LOGIT_CLIP = 20.0
MAX_GRAD_NORM = 1.0


def _to_float_reward(reward):
    if isinstance(reward, (list, tuple, np.ndarray)):
        if len(reward) == 0:
            return 0.0
        value = float(reward[0])
    else:
        value = float(reward)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -REWARD_CLIP, REWARD_CLIP))


def _to_numpy_state(state):
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=STATE_CLIP, neginf=-STATE_CLIP)
    return np.clip(arr, -STATE_CLIP, STATE_CLIP).astype(np.float32, copy=False)


def _sanitize_tensor(x, clip_value):
    x = torch.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return torch.clamp(x, -clip_value, clip_value)


def _safe_logits(logits):
    return _sanitize_tensor(logits, LOGIT_CLIP)


def _normalize_advantages(advantages):
    advantages = _sanitize_tensor(advantages, REWARD_CLIP)
    std = advantages.std(unbiased=False)
    if not torch.isfinite(std) or std.item() < 1e-8:
        return advantages - advantages.mean()
    return (advantages - advantages.mean()) / (std + 1e-8)


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        h1 = in_dim + max(8, int(in_dim * 0.3))
        h2 = max(16, int(in_dim * 0.6))
        h3 = max(8, int(in_dim * 0.2))
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.SELU(),
            nn.Linear(h1, h2),
            nn.SELU(),
            nn.Linear(h2, h3),
            nn.ELU(),
            nn.Linear(h3, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PPOAgent(nn.Module):
    global_memory = deque(maxlen=ppo_cfg["maxlen_mem"])

    def __init__(self, state_size, action_size, checkpoint_path="./", load_model=False):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = ppo_cfg["discount_factor"]
        self.eps_clip = ppo_cfg["eps_clip"]
        self.entropy_coef = ppo_cfg["entropy_coef"]
        self.value_coef = ppo_cfg["value_coef"]
        self.k_epochs = ppo_cfg["k_epochs"]
        self.batch_size = ppo_cfg["batch_size"]
        self.train_start = self.batch_size
        self.memory = deque(maxlen=ppo_cfg["maxlen_mem"])
        self.global_memory = PPOAgent.global_memory
        self.model_file = checkpoint_path
        self.generator = np.random.default_rng(GLOBAL_SEED)
        self.epsilon = 0.0
        self.epsilon_decay = 1.0
        self.epsilon_min = 0.0

        self.actor = _MLP(state_size, action_size).to(device)
        self.critic = _MLP(state_size, 1).to(device)
        self.actor_old = _MLP(state_size, action_size).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=ppo_cfg["learning_rate"])
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=ppo_cfg["learning_rate"])

        if load_model:
            self.load_state_dict(torch.load(self.model_file, map_location=device))

        self.lock = Lock()

    def save_model(self, name):
        torch.save(self.state_dict(), name)

    def update_target_model(self):
        self.actor_old.load_state_dict(self.actor.state_dict())

    def get_actions(self, state, vid):
        state = _sanitize_tensor(state.to(device), STATE_CLIP)
        with torch.no_grad():
            logits = _safe_logits(self.actor(state))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        one_hot = torch.zeros(self.action_size, dtype=torch.float32, device=logits.device)
        one_hot[action.item()] = 1.0
        return [vid, one_hot.unsqueeze(0).detach().cpu()], None

    def add_memory(self, state, action, reward, next_state, done=0):
        if action == -1:
            return
        self.memory.append((
            _to_numpy_state(state),
            int(action),
            _to_float_reward(reward),
            _to_numpy_state(next_state),
            float(done),
        ))

    def add_global_memory(self, state, action, reward, next_state, done=0):
        if action == -1:
            return
        self.global_memory.append((
            _to_numpy_state(state),
            int(action),
            _to_float_reward(reward),
            _to_numpy_state(next_state),
            float(done),
        ))

    def train_model(self):
        if eval_mode:
            return
        if len(self.memory) < self.batch_size:
            return
        self.lock.acquire()
        try:
            source = self.global_memory if (self.generator.random() < ppo_cfg["combine"] and len(self.global_memory) >= self.batch_size) else self.memory
            mini_batch = random.sample(source, self.batch_size)

            states = torch.as_tensor(np.array([b[0] for b in mini_batch]), dtype=torch.float32, device=device)
            actions = torch.as_tensor(np.array([b[1] for b in mini_batch]), dtype=torch.long, device=device)
            rewards = torch.as_tensor(np.array([b[2] for b in mini_batch]), dtype=torch.float32, device=device)
            next_states = torch.as_tensor(np.array([b[3] for b in mini_batch]), dtype=torch.float32, device=device)
            dones = torch.as_tensor(np.array([b[4] for b in mini_batch]), dtype=torch.float32, device=device)

            states = _sanitize_tensor(states, STATE_CLIP)
            next_states = _sanitize_tensor(next_states, STATE_CLIP)
            rewards = _sanitize_tensor(rewards, REWARD_CLIP)
            dones = _sanitize_tensor(dones, 1.0)
            actions = torch.clamp(actions, 0, self.action_size - 1)

            with torch.no_grad():
                values = _sanitize_tensor(self.critic(states).squeeze(-1), REWARD_CLIP)
                next_values = _sanitize_tensor(self.critic(next_states).squeeze(-1), REWARD_CLIP)
                td_target = rewards + self.gamma * (1.0 - dones) * next_values
                advantages = td_target - values
                advantages = _normalize_advantages(advantages)

                old_logits = _safe_logits(self.actor_old(states))
                old_dist = torch.distributions.Categorical(logits=old_logits)
                old_log_probs = old_dist.log_prob(actions)
                if not torch.isfinite(old_log_probs).all():
                    return

            for _ in range(self.k_epochs):
                logits = _safe_logits(self.actor(states))
                if not torch.isfinite(logits).all():
                    return
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                value_pred = _sanitize_tensor(self.critic(states).squeeze(-1), REWARD_CLIP)
                critic_loss = F.mse_loss(value_pred, td_target)
                if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                    return

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                (self.value_coef * critic_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
                self.critic_optim.step()

            self.update_target_model()
        finally:
            self.lock.release()


class A2CAgent(nn.Module):
    global_memory = deque(maxlen=a2c_cfg["maxlen_mem"])

    def __init__(self, state_size, action_size, checkpoint_path="./", load_model=False):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = a2c_cfg["discount_factor"]
        self.entropy_coef = a2c_cfg["entropy_coef"]
        self.value_coef = a2c_cfg["value_coef"]
        self.batch_size = a2c_cfg["batch_size"]
        self.train_start = self.batch_size
        self.memory = deque(maxlen=a2c_cfg["maxlen_mem"])
        self.global_memory = A2CAgent.global_memory
        self.model_file = checkpoint_path
        self.generator = np.random.default_rng(GLOBAL_SEED)
        self.epsilon = 0.0
        self.epsilon_decay = 1.0
        self.epsilon_min = 0.0

        self.actor = _MLP(state_size, action_size).to(device)
        self.critic = _MLP(state_size, 1).to(device)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=a2c_cfg["learning_rate"])
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=a2c_cfg["learning_rate"])

        if load_model:
            self.load_state_dict(torch.load(self.model_file, map_location=device))

        self.lock = Lock()

    def save_model(self, name):
        torch.save(self.state_dict(), name)

    def update_target_model(self):
        return

    def get_actions(self, state, vid):
        state = _sanitize_tensor(state.to(device), STATE_CLIP)
        with torch.no_grad():
            logits = _safe_logits(self.actor(state))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        one_hot = torch.zeros(self.action_size, dtype=torch.float32, device=logits.device)
        one_hot[action.item()] = 1.0
        return [vid, one_hot.unsqueeze(0).detach().cpu()], None

    def add_memory(self, state, action, reward, next_state, done=0):
        if action == -1:
            return
        self.memory.append((
            _to_numpy_state(state),
            int(action),
            _to_float_reward(reward),
            _to_numpy_state(next_state),
            float(done),
        ))

    def add_global_memory(self, state, action, reward, next_state, done=0):
        if action == -1:
            return
        self.global_memory.append((
            _to_numpy_state(state),
            int(action),
            _to_float_reward(reward),
            _to_numpy_state(next_state),
            float(done),
        ))

    def train_model(self):
        if eval_mode:
            return
        if len(self.memory) < self.batch_size:
            return
        self.lock.acquire()
        try:
            source = self.global_memory if (self.generator.random() < a2c_cfg["combine"] and len(self.global_memory) >= self.batch_size) else self.memory
            mini_batch = random.sample(source, self.batch_size)

            states = torch.as_tensor(np.array([b[0] for b in mini_batch]), dtype=torch.float32, device=device)
            actions = torch.as_tensor(np.array([b[1] for b in mini_batch]), dtype=torch.long, device=device)
            rewards = torch.as_tensor(np.array([b[2] for b in mini_batch]), dtype=torch.float32, device=device)
            next_states = torch.as_tensor(np.array([b[3] for b in mini_batch]), dtype=torch.float32, device=device)
            dones = torch.as_tensor(np.array([b[4] for b in mini_batch]), dtype=torch.float32, device=device)

            states = _sanitize_tensor(states, STATE_CLIP)
            next_states = _sanitize_tensor(next_states, STATE_CLIP)
            rewards = _sanitize_tensor(rewards, REWARD_CLIP)
            dones = _sanitize_tensor(dones, 1.0)
            actions = torch.clamp(actions, 0, self.action_size - 1)

            values = _sanitize_tensor(self.critic(states).squeeze(-1), REWARD_CLIP)
            with torch.no_grad():
                next_values = _sanitize_tensor(self.critic(next_states).squeeze(-1), REWARD_CLIP)
                td_target = rewards + self.gamma * (1.0 - dones) * next_values
                advantages = td_target - values
                advantages = _normalize_advantages(advantages)
            logits = _safe_logits(self.actor(states))
            if not torch.isfinite(logits).all():
                return
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
            critic_loss = self.value_coef * F.mse_loss(values, td_target)
            if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                return

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.critic_optim.step()
        finally:
            self.lock.release()


class DDPGAgent(nn.Module):
    global_memory = deque(maxlen=ddpg_cfg["maxlen_mem"])

    def __init__(self, state_size, action_size, checkpoint_path="./", load_model=False):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = ddpg_cfg["discount_factor"]
        self.tau = ddpg_cfg["tau"]
        self.noise_std = ddpg_cfg["noise_std"]
        self.batch_size = ddpg_cfg["batch_size"]
        self.train_start = self.batch_size
        self.memory = deque(maxlen=ddpg_cfg["maxlen_mem"])
        self.global_memory = DDPGAgent.global_memory
        self.model_file = checkpoint_path
        self.generator = np.random.default_rng(GLOBAL_SEED)
        self.epsilon = 0.0
        self.epsilon_decay = 1.0
        self.epsilon_min = 0.0

        self.actor = _MLP(state_size, action_size).to(device)
        self.actor_target = _MLP(state_size, action_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = _MLP(state_size + action_size, 1).to(device)
        self.critic_target = _MLP(state_size + action_size, 1).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=ddpg_cfg["actor_lr"])
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=ddpg_cfg["critic_lr"])

        if load_model:
            self.load_state_dict(torch.load(self.model_file, map_location=device))

        self.lock = Lock()

    def save_model(self, name):
        torch.save(self.state_dict(), name)

    def _soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_((1.0 - self.tau) * t_param.data + self.tau * s_param.data)

    def update_target_model(self):
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _actor_action_onehot(self, states, use_target=False):
        actor = self.actor_target if use_target else self.actor
        logits = _safe_logits(actor(states))
        idx = torch.argmax(logits, dim=1)
        return F.one_hot(idx, num_classes=self.action_size).float()

    def get_actions(self, state, vid):
        state = _sanitize_tensor(state.to(device), STATE_CLIP)
        with torch.no_grad():
            logits = _safe_logits(self.actor(state).squeeze(0))
        noise = torch.normal(0, self.noise_std, size=logits.shape, device=logits.device)
        action_scores = logits + noise
        action = int(torch.argmax(action_scores).item())
        one_hot = torch.zeros(self.action_size, dtype=torch.float32, device=logits.device)
        one_hot[action] = 1.0
        return [vid, one_hot.unsqueeze(0).detach().cpu()], None

    def add_memory(self, state, action, reward, next_state, done=0):
        if action == -1:
            return
        self.memory.append((
            _to_numpy_state(state),
            int(action),
            _to_float_reward(reward),
            _to_numpy_state(next_state),
            float(done),
        ))

    def add_global_memory(self, state, action, reward, next_state, done=0):
        if action == -1:
            return
        self.global_memory.append((
            _to_numpy_state(state),
            int(action),
            _to_float_reward(reward),
            _to_numpy_state(next_state),
            float(done),
        ))

    def train_model(self):
        if eval_mode:
            return
        if len(self.memory) < self.batch_size:
            return
        self.lock.acquire()
        try:
            source = self.global_memory if (self.generator.random() < ddpg_cfg["combine"] and len(self.global_memory) >= self.batch_size) else self.memory
            mini_batch = random.sample(source, self.batch_size)

            states = torch.as_tensor(np.array([b[0] for b in mini_batch]), dtype=torch.float32, device=device)
            actions = torch.as_tensor(np.array([b[1] for b in mini_batch]), dtype=torch.long, device=device)
            rewards = torch.as_tensor(np.array([b[2] for b in mini_batch]), dtype=torch.float32, device=device).unsqueeze(-1)
            next_states = torch.as_tensor(np.array([b[3] for b in mini_batch]), dtype=torch.float32, device=device)
            dones = torch.as_tensor(np.array([b[4] for b in mini_batch]), dtype=torch.float32, device=device).unsqueeze(-1)

            states = _sanitize_tensor(states, STATE_CLIP)
            next_states = _sanitize_tensor(next_states, STATE_CLIP)
            rewards = _sanitize_tensor(rewards, REWARD_CLIP)
            dones = _sanitize_tensor(dones, 1.0)
            actions = torch.clamp(actions, 0, self.action_size - 1)

            action_one_hot = F.one_hot(actions, num_classes=self.action_size).float()

            with torch.no_grad():
                next_action_one_hot = self._actor_action_onehot(next_states, use_target=True)
                target_q = self.critic_target(torch.cat([next_states, next_action_one_hot], dim=1))
                y = rewards + self.gamma * (1.0 - dones) * target_q

            q = self.critic(torch.cat([states, action_one_hot], dim=1))
            critic_loss = F.mse_loss(q, y)
            if not torch.isfinite(critic_loss):
                return

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.critic_optim.step()

            pred_action_one_hot = self._actor_action_onehot(states)
            actor_loss = -self.critic(torch.cat([states, pred_action_one_hot], dim=1)).mean()
            if not torch.isfinite(actor_loss):
                return

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor_optim.step()

            self.update_target_model()
        finally:
            self.lock.release()
