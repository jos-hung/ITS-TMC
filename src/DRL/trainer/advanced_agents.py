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


MAX_GRAD_NORM = 1.0
LOGIT_CLIP = 10.0


def _to_float_reward(reward):
    if isinstance(reward, (list, tuple, np.ndarray)):
        if len(reward) == 0:
            return 0.0
        return float(reward[0])
    return float(reward)


def _to_numpy_state(state):
    if isinstance(state, torch.Tensor):
        arr = state.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(arr)):
        print(f"Warning: Non-finite values in state, replacing with 0")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)


def _normalize_advantages(advantages):
    mean = advantages.mean()
    std = advantages.std()
    if std < 1e-8:
        return advantages - mean
    return (advantages - mean) / (std + 1e-8)


# class _MLP(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         h1 = in_dim + max(8, int(in_dim * 0.3))
#         h2 = max(16, int(in_dim * 0.6))
#         h3 = max(8, int(in_dim * 0.2))
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, h1),
#             nn.Tanh(),
#             nn.Linear(h1, h2),
#             nn.Tanh(),
#             nn.Linear(h2, h3),
#             nn.Tanh(),
#             nn.Linear(h3, out_dim),
#         )

#     def forward(self, x):
#         return self.net(x)

class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logit_scale = 30.0 #hardcoded logit scale to prevent overflow, can be tuned if needed
        
        hidden =in_dim+ max(128, int(in_dim * 0.3))

        self.input = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
        )

        self.block1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

        self.output = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = self.input(x)

        x = x + 0.1 * self.block1(x)
        x = x + 0.1 * self.block2(x)
        raw_logits = self.output(x)
        logits = self.logit_scale * torch.tanh(raw_logits / self.logit_scale)
        return logits


class PPOAgent(nn.Module):
    global_memory = deque(maxlen=a2c_cfg["maxlen_mem"])
    
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

        self.update_frequency = self.batch_size * 4
        self.train_start = self.batch_size

        self.memory = deque(maxlen=ppo_cfg["maxlen_mem"])
        self.model_file = checkpoint_path

        self.actor = _MLP(state_size, action_size).to(device)
        self.critic = _MLP(state_size, 1).to(device)

        self.actor_optim = optim.AdamW(
            self.actor.parameters(),
            lr=ppo_cfg["learning_rate"]
        )

        self.critic_optim = optim.AdamW(
            self.critic.parameters(),
            lr=ppo_cfg["learning_rate"]
        )

        if load_model:
            self.load_state_dict(torch.load(self.model_file, map_location=device))

        self.lock = Lock()

    def save_model(self, name):
        torch.save(self.state_dict(), name)

    def get_actions(self, state, vid):
        state = state.to(device)

        with torch.no_grad():
            logits = self.actor(state)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)

        return [vid, action.cpu().detach()], log_prob.cpu().detach()

    def add_memory(self, state, action, reward, next_state, done=0):
        if action == -1:
            return

        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state_t)
            dist = torch.distributions.Categorical(logits=logits)

            action_t = torch.as_tensor(action, dtype=torch.long, device=device)

            if action_t.dim() == 0:
                old_log_prob = dist.log_prob(action_t.unsqueeze(0)).squeeze(0)
            else:
                old_log_prob = dist.log_prob(action_t)

        self.memory.append((
            _to_numpy_state(state),
            int(action),
            _to_float_reward(reward),
            _to_numpy_state(next_state),
            float(done),
            float(old_log_prob.cpu().item()),
        ))
        
    def add_global_memory(self, state, action, reward, next_state, done=0):
        if action == -1:
            return

        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        with torch.no_grad():
            logits = self.actor(state_t)
            dist = torch.distributions.Categorical(logits=logits)

            action_t = torch.as_tensor(action, dtype=torch.long, device=device)

            if action_t.dim() == 0:
                old_log_prob = dist.log_prob(action_t.unsqueeze(0)).squeeze(0)
            else:
                old_log_prob = dist.log_prob(action_t)

        self.global_memory.append((
            _to_numpy_state(state),
            int(action),
            _to_float_reward(reward),
            _to_numpy_state(next_state),
            float(done),
            float(old_log_prob.cpu().item()),
        ))

    def train_model(self):
        if eval_mode:
            return

        if len(self.memory) < self.update_frequency:
            return

        self.lock.acquire()

        try:
            batch = list(self.memory)

            states = torch.as_tensor(
                np.array([b[0] for b in batch]),
                dtype=torch.float32,
                device=device
            )

            actions = torch.as_tensor(
                np.array([b[1] for b in batch]),
                dtype=torch.long,
                device=device
            )

            rewards = torch.as_tensor(
                np.array([b[2] for b in batch]),
                dtype=torch.float32,
                device=device
            )

            next_states = torch.as_tensor(
                np.array([b[3] for b in batch]),
                dtype=torch.float32,
                device=device
            )

            dones = torch.as_tensor(
                np.array([b[4] for b in batch]),
                dtype=torch.float32,
                device=device
            )

            old_log_probs = torch.as_tensor(
                np.array([b[5] for b in batch]),
                dtype=torch.float32,
                device=device
            )

            actions = torch.clamp(actions, 0, self.action_size - 1)

            with torch.no_grad():
                values = self.critic(states).squeeze(-1)
                next_values = self.critic(next_states).squeeze(-1)

                returns = rewards + self.gamma * (1.0 - dones) * next_values
                advantages = returns - values
                advantages = _normalize_advantages(advantages)

            dataset_size = states.size(0)

            for epoch in range(self.k_epochs):
                indices = torch.randperm(dataset_size, device=device)

                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    idx = indices[start:end]

                    mb_states = states[idx]
                    mb_actions = actions[idx]
                    mb_old_log_probs = old_log_probs[idx]
                    mb_returns = returns[idx]
                    mb_advantages = advantages[idx]

                    logits = self.actor(mb_states)

                    if not torch.isfinite(logits).all():
                        print("Bad logits detected")
                        print("min:", logits.nan_to_num().min().item())
                        print("max:", logits.nan_to_num().max().item())
                        raise RuntimeError("NaN/Inf logits")

                    dist = torch.distributions.Categorical(logits=logits)

                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    log_ratio = new_log_probs - mb_old_log_probs
                    log_ratio = torch.clamp(log_ratio, -20.0, 20.0)

                    ratio = torch.exp(log_ratio)

                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.eps_clip,
                        1.0 + self.eps_clip
                    ) * mb_advantages

                    actor_loss = -torch.min(surr1, surr2).mean()
                    actor_loss = actor_loss - self.entropy_coef * entropy

                    value_pred = self.critic(mb_states).squeeze(-1)
                    critic_loss = F.mse_loss(value_pred, mb_returns)

                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        MAX_GRAD_NORM
                    )
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_total_loss = self.value_coef * critic_loss
                    critic_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        MAX_GRAD_NORM
                    )
                    self.critic_optim.step()

            self.memory.clear()
            print("PPO updated, memory cleared")

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
        state = state.to(device)
        with torch.no_grad():
            logits = self.actor(state)
        return [vid, logits.cpu().detach()], None

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

            actions = torch.clamp(actions, 0, self.action_size - 1)

            values = self.critic(states).squeeze(-1)
            with torch.no_grad():
                next_values = self.critic(next_states).squeeze(-1)
                td_target = rewards + self.gamma * (1.0 - dones) * next_values
                advantages = td_target - values
                advantages = _normalize_advantages(advantages)
            logits = self.actor(states)
            # logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)
            try:
                dist = torch.distributions.Categorical(logits=logits)
            except Exception as e:
                print(f"Error creating distribution: {e}")
                print(f"Logits: {logits}")
                logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)
                dist = torch.distributions.Categorical(logits=logits)
                
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
            critic_loss = self.value_coef * F.mse_loss(values, td_target)

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
        logits = actor(states)
        logits = torch.clamp(logits, -LOGIT_CLIP, LOGIT_CLIP)
        idx = torch.argmax(logits, dim=1)
        return F.one_hot(idx, num_classes=self.action_size).float()

    def get_actions(self, state, vid):
        state = state.to(device)
        with torch.no_grad():
            logits = self.actor(state)
        return [vid, logits.cpu().detach()], None

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

            actions = torch.clamp(actions, 0, self.action_size - 1)

            action_one_hot = F.one_hot(actions, num_classes=self.action_size).float()

            with torch.no_grad():
                next_action_one_hot = self._actor_action_onehot(next_states, use_target=True)
                target_q = self.critic_target(torch.cat([next_states, next_action_one_hot], dim=1))
                y = rewards + self.gamma * (1.0 - dones) * target_q

            q = self.critic(torch.cat([states, action_one_hot], dim=1))
            critic_loss = F.mse_loss(q, y)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.critic_optim.step()

            pred_action_one_hot = self._actor_action_onehot(states)
            actor_loss = -self.critic(torch.cat([states, pred_action_one_hot], dim=1)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor_optim.step()

            self.update_target_model()
        finally:
            self.lock.release()
