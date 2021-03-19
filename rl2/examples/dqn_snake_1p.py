from tensorboard.compat.proto.event_pb2 import TaggedRunMetadata
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

env = gym.make('Snake-v1',
               num_snakes=1, num_fruits=1,
               width=10, height=10,
               vision_range=5)
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
    'buffer_size': int(1e6),
    'batch_size': 64,
    'num_epochs': 1,
    'update_interval': 100000,
    'train_interval': 1,
    'log_interval': 10,
    'lr': 1e-4,
    'gamma': 0.99,
    'eps': 0.1,
    'polyak': 0,
    'decay_step': 1000000,
    'grad_clip': 1,
    'log_dir': './runs',
    'tag': 'DQN/SNAKE/VR',
    'double': False,
    'log_level': 10,
    # 'optim_args': {}
}
config = EasyDict(myconfig)


if __name__ == '__main__':
    logger = Logger(name='DEFAULT', args=config)
    # logger.config_summary(myconfig)
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
                     #  optim_args=config.optim_args,
                     )

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
                     )

    worker = EpisodicWorker(env=env,
                            n_env=1,
                            agent=agent,
                            training=True,
                            max_episodes=1e9,
                            max_steps_per_ep=1e4,
                            log_interval=config.log_interval,
                            render=True,
                            logger=logger,
                            is_save=True,
                            render_mode='rgb_array',
                            )

    worker.run()
