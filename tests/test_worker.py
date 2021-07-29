import pickle
from logging import Logger
from tempfile import TemporaryFile

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


def random_agent_cls(action_space):
    class RandomAgent(Agent):
        def __init__(self, model):
            self.model = model
            self.num_envs = 2

        def act(self, x):
            action = action_space.sample()
            return action_space.sample()

        def collect(self, state, action, reward, done, next_state):
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

    return RandomModel


env = gym.make('CartPole-v0')
model = test_model_cls()
agent_cls = random_agent_cls(env.action_space)
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
    elength = (erange[1] - erange[0])
    worker = CustomWorker(env, agent, 1000, save_erange=erange)
    with worker:
        worker.run()

    with TemporaryFile() as fp:
        pickle.dump(worker.trajectories, fp)
        fp.seek(0)
        data = pickle.load(fp)
        assert len(data) == elength