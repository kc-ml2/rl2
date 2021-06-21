import pickle
from logging import Logger
from pprint import pprint
from tempfile import TemporaryFile, NamedTemporaryFile

import numpy as np
import pytest
import gym
from torch import nn

from rl2.agents.base import Agent
from rl2.workers import RolloutWorker

logger = Logger(__name__)


class CustomWorker(RolloutWorker):
    def __init__(self, env, agent, max_steps, **kwargs):
        super(CustomWorker, self).__init__(env, agent, **kwargs)
        self.max_steps = max_steps

    def run(self):
        for step in range(self.max_steps):
            self.rollout()


def test_agent_cls(action_space):
    class RandomAgent:
        def __init__(self, model):
            self.model = model

        def act(self, x):
            action = action_space.sample()
            logger.info(action)
            return action_space.sample()

        def collect(self, s, a, r, d, s_):
            pass

        def train(self):
            pass

        def step(self, *args):
            pass

    return RandomAgent


def test_model_cls():
    class RandomModel(nn.Module):
        def __init__(self):
            super(RandomModel, self).__init__()

        # def forward(self, x):

    return RandomModel


env = gym.make('CartPole-v0')
model = test_model_cls()
agent_cls = test_agent_cls(env.action_space)
agent = agent_cls(model)


def test_run():
    worker = CustomWorker(env, agent, 10)
    worker.run()


def test_context():
    worker = CustomWorker(env, agent, 10)
    with worker:
        worker.run()


def test_save_trajectories():
    # save from episode 10 to 30, exclusive
    erange = (10, 31)
    elength = (erange[1] - erange[0]) - 1
    worker = CustomWorker(env, agent, 1000, save_erange=(10, 31))
    with worker:
        worker.run()

    with TemporaryFile() as fp:
        # filename = 'expert_data.pickle'
        # with open(fp.name, 'wb') as fp:
        pickle.dump(worker.trajectories, fp)
        # print(worker.trajectories)
        fp.seek(0)
        # with open(filename, 'rb') as fp:
        data = pickle.load(fp)
        pprint(data)
        assert len(data) == elength