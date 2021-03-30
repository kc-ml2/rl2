from rl2.agents.utils import LinearDecay
from rl2.examples.temp_logger import LOG_LEVELS, Logger
from rl2.agents.dqn import DQNAgent, DQNModel
import gym
import marlenv
from easydict import EasyDict
from rl2.workers.multi_agent import IndividualEpisodicWorker

# FIXME: Remove later
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

myconfig = {
    'buffer_size': 1000000,
    'batch_size': 1024,
    'num_epochs': 1,
    'update_interval': 100000,
    'train_interval': 1,
    'log_interval': 100,
    'lr': 1e-4,
    'gamma': 0.99,
    'eps_start': 0.5,
    'eps_end': 0.001,
    'polyak': 0,
    'decay_step': 2000000,
    'grad_clip': 10,
    'tag': 'MADQN/pretrained',
    'load_dir': '/home/eunki/rl2/rl2/examples/runs/DDQN/SNAKE/FS2/MS20NF4/20210325163351/ckpt/6000k/DQNModel.pt',
    'double': False,
    'log_level': 10,
    'env_kwargs': {'num_snakes': 4,
                   'num_fruits': 4,
                   'width': 20, 'height': 20,
                   'frame_stack': 2,
                   'vision_range': 5}
}
config = EasyDict(myconfig)

eps = LinearDecay(start=config.eps_start,
                  end=config.eps_end,
                  decay_step=config.decay_step)

env = gym.make('Snake-v1', **config.env_kwargs)

joint_obs_shape = env.observation_space.shape
joint_act_shape = []
for i in range(len(env.action_space.n)):
    joint_act_shape.append((env.action_space.n[i],))

if __name__ == "__main__":
    logger = Logger(name='MADQN', args=config)
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
        model.load(config.load_dir)
        agent = DQNAgent(model,
                         update_interval=config.update_interval,
                         batch_size=config.batch_size,
                         decay_step=config.decay_step,
                         eps=LinearDecay(start=config.eps_start,
                                         end=config.eps_end,
                                         decay_step=config.decay_step),
                         gamma=config.gamma,
                         )
        agents.append(agent)

    worker = IndividualEpisodicWorker(env, agents,
                                      max_episodes=100000,
                                      training=True,
                                      render=True,
                                      logger=logger,
                                      log_interval=config.log_interval,
                                      is_save=True,
                                      )
    worker.run()
