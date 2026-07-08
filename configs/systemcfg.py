"""System-wide configuration defaults.

This module exposes low-level runtime defaults used across the project:
- device and random seed defaults
- logging configuration
- mission/task/map/network defaults
- DRL hyperparameters (e.g. `ddqn_cfg`)

Modules that need runtime constants import values from here. Changing values
will affect runs globally.
"""

DEVICE = 0
GLOBAL_SEED = 42

log_configs = {
    'log_dir': 'logs'
}

apply_thread = 0
apply_detach = 0
score_window_size = 100
tau = 120
mission_cfg = {
    'n_mission': 25,
    'benifits': [50,100],
    'n_vehicle': 5,
    'n_miss_per_vec': 5
}

apply_early_stopping = False #proposed 1 trong paper
apply_offloading_based_max_allow_average_delay = True #proposed 2 trong paper
apply_robots_training = True 

ppo_cfg = {
    'benifits': [50,100],
    'update_frequency': 60,
    'save_dir': 'checkpoints/ppo_more_epoch',
    'max_eps_length': 100,
    'score_window_size': 100,
    'thread': apply_thread,
    'detach_thread': apply_detach,
    'type_': 'MAPPOTrainer',
    'discount_factor': 0.99,
    'learning_rate': 3e-4,
    'batch_size': 128,
    'maxlen_mem': 200000,
    'eps_clip': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'k_epochs': 4,
    'combine': 0.0,
    'modify_reward': True,
    'reward_dep_scale': 50.0,
    'reward_wait_scale': 50.0,
    'reward_completed_scale': 1.0,
    'reward_use_vehicle_bias': False,
    'team_reward_alpha_start': 0.05,
    'team_reward_alpha_end': 0.15,
    'team_reward_alpha_warmup_episodes': 30000,
    'fairness_gap_lambda': 0.0005
}

a2c_cfg = {
    'discount_factor': 0.99,
    'learning_rate': 3e-4,
    'batch_size': 256,
    'maxlen_mem': 200000,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'combine': 0.0,
    'modify_reward': True,
    'reward_dep_scale': 50.0,
    'reward_wait_scale': 50.0,
    'reward_completed_scale': 1.0,
    'reward_use_vehicle_bias': False,
    'team_reward_alpha_start': 0.05,
    'team_reward_alpha_end': 0.15,
    'team_reward_alpha_warmup_episodes': 30000,
    'fairness_gap_lambda': 0.0005
}

ddpg_cfg = {
    'discount_factor': 0.99,
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,
    'tau': 0.005,
    'noise_std': 0.1,
    'batch_size': 256,
    'maxlen_mem': 200000,
    'combine': 0.0,
    'modify_reward': True,
    'reward_dep_scale': 50.0,
    'reward_wait_scale': 50.0,
    'reward_completed_scale': 1.0,
    'reward_use_vehicle_bias': False,
    'team_reward_alpha_start': 0.05,
    'team_reward_alpha_end': 0.15,
    'team_reward_alpha_warmup_episodes': 30000,
    'fairness_gap_lambda': 0.0005
}

    

task_cfg = {
    'comm_size':[100, 500], #kbytes
    'comp_size':[1,3], #mcycles
    'lambdas': [10, 30, 50], #tasks/second
    'vmax' : 10, #m/s
    'tau' : tau, #phut
    'cost_coefi' : 5*10**-5,
    'max_speed': 20 #m/s
    
}
def get_ideal_avg_reward():
    maximum_earned_path = task_cfg['max_speed']*task_cfg['tau']*60 #m
    avg_ben = (mission_cfg['benifits'][0]+mission_cfg['benifits'][1])/2
    length_path_avg = 2000 #m
    number_of_complete = maximum_earned_path/length_path_avg
    return number_of_complete*avg_ben + number_of_complete*100

avg_reward = get_ideal_avg_reward()

map_cfg = {
    'real_map': True,
    # 'real_center_point': (21.007837, 105.841819), #bkhn
    # 'real_center_point': (20.995417, 105.950051), #vin-university
    'real_center_point': (55.7038298,13.1944803), #lund university
    'radius': 2500,
    'n_lines': 15,
    'busy': 1,
    'fromfile': 1
}
network_cfg = {
    "n_MEC":20,
    "CPU_freq": [100, 300],#mcycles
    "CPU_satelite": 50, #mcycles
    "satelite_distance": 1000, #km
    "path_loss": 3,
    "channel_gain": "Gausian",
    "best_rate_radius": 100, #m
    "seed":42 
}
vehicle_cfg = {
    # Per-robot mobility parameters (Paper Section II, Eqs. 13-20)
    'v_nominal': 10.0,   # nominal traveling speed v^0_v (m/s)
    'v_min': 0.0,        # minimum safe speed v^min_v (m/s)
    'rho_down': 2.0,     # safe deceleration rate rho^down_v (m/s^2)
    'rho_up': 2.0,       # acceleration rate rho^up_v (m/s^2)
    'cpu_freqz': 10.0, # CPU capacity f^max_v (mcycles/s)
}

eval = False
ddqn_cfg = {
    "discount_factor":0.95,
    "learning_rate": 1e-5,
    "epsilon": 1.0,
    "epsilon_decay": 0.999,
    "epsilon_min": 0.1,
    "batch_size": 512,
    "maxlen_mem": 10000000,
    "modify_reward": True,
    "combine": 0.0,
    "rotate_selection_order": True,
    "conflict_penalty_scale": 0.01,
    "conflict_opportunity_aware": True,
    "conflict_min_available": 1,
    "reward_dep_scale": 50.0,
    "reward_wait_scale": 50.0,
    "reward_completed_scale": 1.0,
    "reward_use_vehicle_bias": False,
    "team_reward_alpha_start": 0.05,
    "team_reward_alpha_end": 0.15,
    "team_reward_alpha_warmup_episodes": 30000,
    "fairness_gap_lambda": 0.0005
}
