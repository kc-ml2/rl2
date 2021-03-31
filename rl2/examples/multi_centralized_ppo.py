from easydict import EasyDict

import gym
from gym.vector.async_vector_env import AsyncVectorEnv

import marlenv
from marlenv.wrappers import SingleMultiAgent, AsyncVectorMultiEnv

from rl2.examples.temp_logger import LOG_LEVELS, Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.agents.configs import DEFAULT_DQN_CONFIG
from rl2.workers.multi_agent import CentralizedEpisodicWorker, IndividualEpisodicWorker

# FIXME: Remove later
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


num_snakes = 4


def make_snake():
    def _make():
        env = gym.make('Snake-v1',
                       num_snakes=num_snakes, num_fruits=4,
                       width=20, height=20,
                       vision_range=5)
        env = SingleMultiAgent(env)
        return env
    return _make


def make_vec():
    n_env = 64
    dummyenv = make_snake()()
    observation_shape = dummyenv.observation_space.shape[1:]
    action_shape = (dummyenv.action_space.n,)
    env = AsyncVectorMultiEnv([make_snake() for i in range(n_env)])
    high = dummyenv.observation_space.high
    return n_env, env, observation_shape, action_shape, high


n_env, env, observation_shape, action_shape, high = make_vec()
reorder = True


myconfig = {
    'buffer_size': 1000000,
    'batch_size': 1024,
    'num_epochs': 1,
    'update_interval': 100000,
    'train_interval': 1,
    'log_interval': 10,
    'lr': 1e-4,
    'gamma': 0.99,
    'eps': 0.0001,
    'polyak': 0,
    'decay_step': 500000,
    'grad_clip': 10,
    'tag': 'MAPPO/',
    'double': False,
    'log_level': 10,
}
config = EasyDict(myconfig)


def ppo(obs_shape, ac_shape):
    model = PPOModel(obs_shape,
                     ac_shape,
                     recurrent=False,
                     discrete=True,
                     reorder=reorder,
                     optimizer='torch.optim.RMSprop',
                     high=high)
    train_interval = 128
    num_env = n_env
    epoch = 4
    batch_size = 512
    agent = PPOAgent(model,
                     train_interval=train_interval,
                     n_env=n_env*num_snakes,
                     batch_size=batch_size,
                     num_epochs=epoch,
                     buffer_kwargs={'size': train_interval,
                                    'n_env': num_env*num_snakes})
    return agent


if __name__ == "__main__":
    logger = Logger(name='CPPO', args=config)
    agent = ppo(observation_shape, action_shape)
    # agents = [ppo(observation_shape, action_shape)]*4
    # worker = IndividualEpisodicWorker(env, agents,
    #                                   max_episodes=100000,
    #                                   training=True,
    #                                   logger=logger,
    #                                   log_interval=config.log_interval,
    #                                   max_steps_per_ep=10000,
    #                                   n_env=n_env,
    #                                   render=True,
    #                                   )
    worker = CentralizedEpisodicWorker(env, n_env, agent,
                                       max_episodes=100000,
                                       training=True,
                                       logger=logger,
                                       log_interval=config.log_interval,
                                       max_steps_per_ep=10000,
                                       render=True,
                                       )
    worker.run()
