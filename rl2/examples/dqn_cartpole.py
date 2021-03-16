from easydict import EasyDict
import gym
from rl2.agents.dqn import DQNModel, DQNAgent
from rl2.workers import MaxStepWorker
from rl2.examples.temp_logger import Logger


env = gym.make("CartPole-v0")

observation_shape = env.observation_space.shape
action_shape = (env.action_space.n,)
myconfig = {
    'log_dir': './runs',
    'tag': 'DQN/snake',
    'log_level': 10
}

myconfig = EasyDict(myconfig)
logger = Logger(name='DEFAULT', args=myconfig)


def dqn():
    model = DQNModel(observation_shape,
                     action_shape,
                     optim='torch.optim.Adam',
                     discrete=True,
                     default=True)
    agent = DQNAgent(model,
                     buffer_size=100000,
                     update_interval=100,
                     )
    return agent


def ddqn():
    model = DQNModel(observation_shape,
                     action_shape,
                     double=True,)
    agent = DQNAgent(model)
    return agent


def main():
    agent = dqn()
    worker = MaxStepWorker(env, agent, max_steps=int(2e5), training=True)
    worker.run()


if __name__ == '__main__':
    main()
