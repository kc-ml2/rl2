import os
import json
from easydict import EasyDict

import marlenv
from marlenv.wrappers import make_snake
import torch

from rl2.examples.temp_logger import Logger
from rlena.algos.utils import Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers.multi_agent import MAMaxStepWorker, MAEpisodicWorker

# FIXME: Remove later
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def ppo(obs_shape, ac_shape, config, props, load_dir=None):
    model = PPOModel(obs_shape,
                     ac_shape,
                     recurrent=False,
                     discrete=True,
                     reorder=True,#props.reorder,
                     optimizer='torch.optim.RMSprop',
                     high=props.high)
    if load_dir is not None:
        model.load(load_dir)
    agent = PPOAgent(model,
                     train_interval=config.train_interval,
                     #  n_env=props.n_env,
                     n_env=props.num_envs,
                     batch_size=config.batch_size,
                     num_epochs=config.epoch,
                     buffer_kwargs={'size': config.train_interval,
                                    # 'n_env': props.n_env})
                                    'n_env': props.num_envs})
    return agent


def train(config):
    logger = Logger(name='PPOCOOP', args=config)
    env, observation_shape, action_shape, props = make_snake(
        env_id='SnakeCoop-v1',
        num_envs=config.n_env,
        num_snakes=config.num_snakes,
        num_fruits=config.num_snakes * 2,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack,
        reward_dict=config.custom_rewardf,
    )
    props = EasyDict(props)

    agents = []
    for _ in range(config.num_snakes):
        agents.append(ppo(observation_shape, action_shape, config, props))

    worker = MAMaxStepWorker(
        env, props.num_envs, agents,                     
        max_steps=int(1e8),
        training=True,
        logger=logger,
        log_interval=config.log_interval,
        render=False,
        render_interval=500000,
        is_save=True,
        save_interval=config.save_interval,
    )
    # with torch.autograd.set_detect_anomaly(True):
    worker.run()     

    return logger.log_dir


def test(config, load_dir=None):
    # Test phase
    if load_dir is not None:
        config_file = os.path.join(load_dir, "config.json")
        model_dir = os.path.join(load_dir, "ckpt")
        with open(config_file, "r") as config_f:
            _config = EasyDict(json.load(config_f))
    else:
        model_dir = None
    logger = Logger(name='PPOCOOP', args=config)

    env, observation_shape, action_shape, props = make_snake(
        num_envs=1,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack,
        reward_dict=config.custom_rewardf
    )
    agents = []
    for i in range(config.num_snakes):
        if model_dir is not None:
            model_file = os.path.join(model_dir,
                                      f'agent{i}', '100k', 'PPOModel.pt')
        else:
            model_file = None
        agents.append(
            ppo(observation_shape, action_shape, config, props,
                load_dir=model_file)
        )

    worker = MAEpisodicWorker(env, props.num_envs, agents,
                              max_episodes=3, training=False,
                              render=True,
                              render_interval=1,
                              logger=logger)
    worker.run()


if __name__ == "__main__":
    myconfig = {
        'n_env': 24,
        'num_snakes': 2,
        'width': 14,
        'height': 14,
        'vision_range': 5,
        'frame_stack': 2,
        'batch_size': 512,
        'epoch': 4,
        'train_interval': 512,
        'log_level': 10,
        'log_interval': 5000,
        'save_interval': 1000000,
        'lr': 1e-4,
        'gamma': 0.99,
        'grad_clip': 10,
        'tag': 'PPO',
        'custom_rewardf': {
            'fruit': 1.0,
            'kill': -1.0,
            'lose': 0.0,
            'win': 0.0,
            'time': -0.01
        }
    }
    config = EasyDict(myconfig)

    log_dir = train(config)
    
