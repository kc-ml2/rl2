from easydict import EasyDict

import gym
from gym.vector.async_vector_env import AsyncVectorEnv

import marlenv
from marlenv.wrappers import make_snake

from rl2.examples.temp_logger import Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers.multi_agent import MAMaxStepWorker

# FIXME: Remove later
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


myconfig = {
    'n_env': 16,
    'num_snakes': 4,
    'width': 20,
    'height': 20,
    'vision_range': 5,
    'frame_stack': 2,
    'batch_size': 128,
    'epoch': 4,
    'train_interval': 128,
    'log_interval': 20000,
    'lr': 1e-4,
    'gamma': 0.99,
    'grad_clip': 10,
    'tag': 'DEBUG/',
    'double': False,
    'log_level': 10,
}
config = EasyDict(myconfig)


def ppo(obs_shape, ac_shape, config, props):
    model = PPOModel(obs_shape,
                     ac_shape,
                     recurrent=False,
                     discrete=True,
                     reorder=props.reorder,
                     optimizer='torch.optim.RMSprop',
                     high=props.high)
    agent = PPOAgent(model,
                     train_interval=config.train_interval,
                     n_env=props.n_env,
                     batch_size=config.batch_size,
                     num_epochs=config.epoch,
                     buffer_kwargs={'size': config.train_interval,
                                    'n_env': props.n_env})
    return agent


if __name__ == "__main__":
    logger = Logger(name='MATEST', args=config)

    env, observation_shape, action_shape, props = make_snake(
        n_env=config.n_env,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_rang=config.vision_range,
        frame_stack=config.frame_stack
    )

    # import pdb; pdb.set_trace()

    agents = []
    # for i, (obs_shape, act_shape) in enumerate(zip(observation_shape,
    #                                                action_shape)):
    for _ in range(config.num_snakes):
        agents.append(ppo(observation_shape, action_shape, config, props))

    worker = MAMaxStepWorker(env, props.n_env, agents,
                             max_steps=int(1e5),
                             training=True,
                             logger=logger,
                             log_interval=config.log_interval,
                             render=True,
                             )
    worker.run()
