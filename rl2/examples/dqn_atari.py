import os
import json
import numpy as np
from easydict import EasyDict

import gym
from gym.wrappers import AtariPreprocessing, FrameStack

from rl2.agents.dqn import DQNAgent, DQNModel
from rl2.agents.configs import DEFAULT_DQN_CONFIG
from rl2.workers.base import EpisodicWorker, MaxStepWorker
from rl2.examples.temp_logger import Logger
from rl2.networks import DeepMindEnc

# FIXME: Remove later
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class NumpyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return np.array(obs), rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return np.array(obs)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames so it's important to keep lives > 0, so that we only reset
            # once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


def dqn(obs_shape, ac_shape, config, load_dir=None):
    encoder = DeepMindEnc(obs_shape)
    model = DQNModel(observation_shape=obs_shape,
                     action_shape=ac_shape,
                     encoder=encoder,
                     encoded_dim=256,
                     double=config.double,
                     recurrent=config.recurrent,
                     optimizer=config.optimizer,
                     lr=config.lr,
                     grad_clip=config.grad_clip,
                     polyak=config.polyak,
                     reorder=False,
                     discrete=True,
                     high=255.)
    if load_dir is not None:
        model.load(load_dir)
    agent = DQNAgent(model,
                     update_interval=config.update_interval,
                     train_interval=config.train_interval,
                     train_after=config.train_after,
                     num_epochs=config.num_epochs,
                     buffer_size=config.buffer_size,
                     batch_size=config.batch_size,
                     decay_step=config.decay_step,
                     eps=config.eps,
                     gamma=config.gamma,
                     log_interval=config.log_interval,)
    return agent


def train(config):
    logger = Logger(name='DEFAULT', args=config)
    env = gym.make('BreakoutNoFrameskip-v4')
    env = AtariPreprocessing(env)
    # env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = NumpyWrapper(env)
    observation_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)
    agent = dqn(observation_shape, action_shape, config)
    worker = MaxStepWorker(env, agent,
                           max_steps=config.max_step,
                           log_interval=config.log_interval,
                           render_mode='rgb_array',
                           render_interval=500000,
                           save_interval=config.save_interval,
                           logger=logger)

    worker.run()
    return logger.log_dir


def test(config, load_dir=None):
    # Test phase
    if log_dir is not None:
        config_file = os.path.join(load_dir, "config.json")
        model_file = os.path.join(load_dir, "ckpt", "1k", "DQNModel.pt")
        with open(config_file, "r") as config_f:
            config = EasyDict(json.load(config_f))
    logger = Logger(name='TUTORIAL', args=config)

    env = gym.make('BreakoutNoFrameskip-v4')
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)
    env = NumpyWrapper(env)
    observation_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)
    agent = dqn(observation_shape, action_shape, config, load_dir=model_file)
    worker = EpisodicWorker(env=env,
                            agent=agent,
                            max_episodes=3,
                            max_steps_per_ep=1e4,
                            log_interval=config.log_interval,
                            logger=logger,
                            render_mode='rgb_array',
                            render_interval=1,
                            )
    worker.run()


if __name__ == '__main__':
    # Use Default config
    config = DEFAULT_DQN_CONFIG

    # Or Customize your config
    myconfig = {
        'buffer_size': int(1e6),
        'batch_size': 32,
        'num_epochs': 1,
        'max_step': int(1e7),
        'update_interval': int(4e4),
        'train_interval': 4,
        'train_after': 50000,
        'log_interval': 20000,
        'save_interval': int(1e6),
        'optimizer': 'torch.optim.RMSprop',
        'lr': 2.5e-4,
        'recurrent': False,
        'gamma': 0.99,
        'eps': 0.01,
        'polyak': 0,
        'decay_step': int(1e6),
        'grad_clip': 0.5,
        'tag': 'DDQN/breakout/l2',
        'double': True,
        'log_level': 10,
    }
    config = EasyDict(myconfig)

    log_dir = train(config)
    # test(config, load_dir=log_dir)
