from rl2.agents.dqn import DQNAgent, DQNModel
from termcolor import colored
import gym
import marlenv
from marlenv.wrappers import SingleAgent
from easydict import EasyDict
from rl2.agents.configs import DEFAULT_DQN_CONFIG
from rl2.workers.base import EpisodicWorker
from rl2.examples.temp_logger import LOG_LEVELS, Logger

# FIXME: Remove later
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


env = gym.make('Snake-v1',
               num_snakes=1, num_fruits=4,
               width=20, height=20,
               frame_stack=4,
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
    'buffer_size': 1000000,
    'batch_size': 1024,
    'num_epochs': 1,
    'update_interval': 100000,
    'train_interval': 1,
    'log_interval': 100,
    'lr': 1e-4,
    'gamma': 0.99,
    'eps': 0.0001,
    'polyak': 0,
    'decay_step': 1000000,
    'grad_clip': 10,
    'tag': 'DDQN/SNAKE/FS4/MS20NF4',
    'double': True,
    'log_level': 10,
}
config = EasyDict(myconfig)


if __name__ == '__main__':
    logger = Logger(name='DEFAULT', args=config)
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
                     )

    agent = DQNAgent(model,
                     update_interval=config.update_interval,
                     train_interval=config.train_interval,
                     num_epochs=config.num_epochs,
                     buffer_size=config.buffer_size,
                     batch_size=config.batch_size,
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
                            render_interval=10000,
                            )

    worker.run()
