from rl2.examples.temp_logger import LOG_LEVELS, Logger
from rl2.agents.dqn import DQNAgent, DQNModel
from rl2.agents.base import MAgent
import gym
import marlenv
from easydict import EasyDict
from rl2.agents.configs import DEFAULT_DQN_CONFIG
from rl2.workers.multi_agent import EpisodicWorker, IndividualEpisodicWorker, SelfMaxStepWorker, MAMaxStepWorker

# FIXME: Remove later
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


env = gym.make('Snake-v1',
               num_snakes=4, num_fruits=4,
               width=20, height=20,
               vision_range=5)

joint_obs_shape = env.observation_space.shape
joint_act_shape = []
for i in range(len(env.action_space.n)):
    joint_act_shape.append((env.action_space.n[i],))

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
    'tag': 'MADQN/',
    'double': False,
    'log_level': 10,
}
config = EasyDict(myconfig)


if __name__ == "__main__":
    logger = Logger(name='MATEST', args=config)
    agents = []
    for i, (obs_shape, act_shape) in enumerate(zip(joint_obs_shape, joint_act_shape)):
        model = DQNModel(obs_shape,
                         act_shape,
                         double=config.double,
                         lr=config.lr,
                         grad_clip=config.grad_clip,
                         polyak=config.polyak,
                         reorder=True,
                         discrete=True,
                         )
        agent = DQNAgent(model,
                         update_interval=config.update_interval,
                         batch_size=config.batch_size,
                         decay_step=config.decay_step,
                         eps=config.eps,
                         gamma=config.gamma,
                         )
        agents.append(agent)
    worker = IndividualEpisodicWorker(env, agents,
                                      max_episodes=100000,
                                      training=True,
                                      render=True,
                                      logger=logger,
                                      log_interval=config.log_interval,
                                      )
    worker.run()

    # worker = EpisodicWorker(env)
