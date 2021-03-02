DEFAULT_PPO_CONFIG = {
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'clip_param': 0.9,
    'lr': 2.5e-4,
    'batch_size': 512,
    'num_epochs': 4,
}

DEFAULT_DQN_CONFIG = {
    'num_workers': 1,
    'batch_size': 32,
    'num_epochs': 1,
    'update_interval': 4,
    'lr': 1.0e-4,
    'buffer_size': int(1e6),
    'init_collect': 20000,
    'target_update': 40000,
}

DEFAULT_DUELINGDQN_CONFIG = {
    'num_workers': 1,
    'batch_size': 32,
    'num_epochs': 1,
    'update_interval': 4,
    'lr': 1.0e-4,
    'buffer_size': int(1e6),
    'init_collect': 20000,
    'target_update': 40000,
}

DEFAULT_A2C_CONFIG = {
    'num_workers': 64,
    'batch_size': 256,
    'num_epochs': 4,
    'update_interval': 16,
    'lr': 1e-3,
    'vf_coef': 0.5,
    'ent_coef': 0.01,
}

DEFAULT_DDPG_CONFIG = {
    'num_workers': 64,
    'buffer_size': int(1e5),
    'batch_size': 32,
    'num_epochs': 1,
    'update_interval': 10,
    'train_interval': 1,
    'log_interval': 1000,
    'lr_ac': 1e-4,
    'lr_cr': 1e-4,
    'gamma': 0.99,
    'eps': 0.00001,
    'polyak': 0.99,
    'grad_clip': 0.1
}
