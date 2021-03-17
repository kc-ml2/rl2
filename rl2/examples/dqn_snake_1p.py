from rl2.agents.dqn import DQNAgent, DQNModel
import torch
from termcolor import colored
import json
import gym
import marlenv
from marlenv.wrappers import SingleAgent
from easydict import EasyDict
from rl2.agents.configs import DEFAULT_DQN_CONFIG
from rl2.workers.base import EpisodicWorker

"""
you might want to modify
1. layer architecture -> just pass nn.Module to predefined models
2. which distributions to use -> implement model from interfaces e.g. implement ActorCritic for custom PPO
3. how to sample distributions -> customize Agent
etc...
below example just changes 1. and some hparams
"""

from rl2.examples.temp_logger import LOG_LEVELS, Logger

env = gym.make('Snake-v1', num_snakes=1, num_fruits=20)
env = SingleAgent(env)

# check Continuous or Discrete
if 'Discrete' in str(type(env.action_space)):
    action_n = env.action_space.n

if 'Box' in str(type(env.action_space)):
    action_low = env.action_space.low
    action_high = env.action_space.high

# Use Default config
config = DEFAULT_DQN_CONFIG

# Or Customize your config
myconfig = {
    # 'num_workers': 64,
    'buffer_size': int(1e6),
    'batch_size': 32,
    'num_epochs': 1,
    'update_interval': 10000,
    'train_interval': 1,
    'log_interval': 10,
    'lr': 1e-4,
    'gamma': 0.99,
    'eps': 0.05,
    'polyak': 0,
    'decay_step': 100000,
    'grad_clip': 0.01,
    'log_dir': './runs',
    'tag': 'DQN/SNAKE',
    'double': False,
    'log_level': 10
}
config = EasyDict(myconfig)

if __name__ == '__main__':
    logger = Logger(name='DEFAULT', args=config)
    import json
    with open(logger.log_dir+'/config.json', 'w') as f:
        json.dump(myconfig, f)
    observation_shape = env.observation_space.shape
    action_shape = (env.action_space.n,) if hasattr(
        env.action_space, 'n') else env.action_space.shape

    model = DQNModel(observation_shape=observation_shape,
                     action_shape=action_shape,
                     double=config.double,
                     lr=config.lr,
                     grad_clip=config.grad_clip,
                     polyak=config.polyak,
                     reorder=True,
                     discrete=True,
                     is_save=True)

    agent = DQNAgent(model,
                     action_n=action_n,
                     update_interval=config.update_interval,
                     train_interval=config.train_interval,
                     num_epochs=config.num_epochs,
                     buffer_size=config.buffer_size,
                     decay_step=config.decay_step,
                     eps=config.eps,
                     gamma=config.gamma,
                     log_interval=config.log_interval,
                     logger=logger
                     )

    worker = EpisodicWorker(env=env,
                            agent=agent,
                            training=True,
                            max_episodes=1e9,
                            max_steps_per_ep=1e4,
                            log_interval=config.log_interval,
                            render=False,
                            logger=logger,
                            )

    worker.run()
