import gym

from rl2.agents import PPOAgent
from rl2.agents.ppo import PPOModel
from rl2.agents.base import Agent
from rl2.models.base import TorchModel
from rl2.networks.core import MLP


class TestAgent(Agent):
    def __init__(self):
        pass

    def act(self):
        pass

    def collect(self, state, action, reward, done, next_state):
        pass

    def train(self):
        pass

    def step(self):
        pass


class TestModel(TorchModel):
    def __init__(self, obs_shape, ac_shape):
        super(TestModel, self).__init__(obs_shape, ac_shape)
        self.policy = MLP(obs_shape, ac_shape)

    def step(self, loss):
        pass

    def save(self):
        pass

    def load(self):
        pass


def test_step():
    env = gym.make('LunarLander-v2')
    obs_shape = env.observation_space.shape
    ac_shape = (env.action_space.n,) if hasattr(
        env.action_space, 'n'
    ) else env.action_space.shape
    obs = env.reset()

    model = PPOModel(obs_shape, ac_shape)
    agent = PPOAgent(model, train_interval=128)

    ac = agent.act(obs)
    next_obs, rew, done, _ = env.step(ac)

    print(agent.step(obs, ac, rew, done, next_obs))
