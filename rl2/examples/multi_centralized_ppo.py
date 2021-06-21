import os
import json
from easydict import EasyDict

import marlenv
from marlenv.wrappers import make_snake

from rl2.examples.temp_logger import Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers.multi_agent import SelfMaxStepWorker, SelfEpisodicWorker

# FIXME: Remove later
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def ppo(obs_shape, ac_shape, config, props, load_dir=None):
    model = PPOModel(obs_shape,
                     ac_shape,
                     recurrent=False,
                     discrete=True,
                     reorder=props.reorder,
                     optimizer='torch.optim.RMSprop',
                     high=props.high)
    if load_dir is not None:
        model.load(load_dir)
    agent = PPOAgent(model,
                     train_interval=config.train_interval,
                     n_env=props.num_envs * config.num_snakes,
                     batch_size=config.batch_size,
                     num_epochs=config.epoch,
                     buffer_kwargs={'size': config.train_interval,
                                    'n_env': props.num_envs * config.num_snakes})
    return agent


def train(config):
    logger = Logger(name='CPPO', args=config)
    env, observation_shape, action_shape, props = make_snake(
        n_env=config.num_envs,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_rang=config.vision_range,
        frame_stack=config.frame_stack
    )

    agent = ppo(observation_shape, action_shape, config, props)

    worker = SelfMaxStepWorker(env, props.num_envs, agent,
                               n_agents=config.num_snakes,
                               max_steps=int(1e7),
                               training=True,
                               logger=logger,
                               log_interval=config.log_interval,
                               render=True,
                               render_interval=500000,
                               is_save=True,
                               save_interval=config.save_interval,
                               )
    worker.run()

    return logger.log_dir


def test(config, load_dir=None):
    # Test phase
    if load_dir is not None:
        config_file = os.path.join(load_dir, "config.json")
        model_file = os.path.join(load_dir, "ckpt", "5000k", "PPOModel.pt")
        with open(config_file, "r") as config_f:
            config = EasyDict(json.load(config_f))
    else:
        model_file = None
    logger = Logger(name='MATEST', args=config)

    env, observation_shape, action_shape, props = make_snake(
        n_env=1,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack
    )

    agent = ppo(observation_shape, action_shape, config, props,
                load_dir=model_file)

    worker = SelfEpisodicWorker(env, 1, agent,
                                n_agents=config.num_snakes,
                                max_episodes=3,
                                training=False,
                                logger=logger,
                                log_interval=config.log_interval,
                                render=True,
                                render_interval=1,
                                is_save=False,
                                )
    worker.run()


if __name__ == "__main__":
    myconfig = {
        'n_env': 64,
        'num_snakes': 4,
        'width': 20,
        'height': 20,
        'vision_range': 5,
        'frame_stack': 2,
        'train_interval': 128,
        'epoch': 4,
        'batch_size': 512,
        'lr': 1e-4,
        'gamma': 0.99,
        'grad_clip': 10,
        'recurrent': False,
        'log_interval': 20000,
        'log_level': 10,
        'save_interval': 1000000,
        'tag': 'TUTORIAL/CPPO',
    }
    config = EasyDict(myconfig)

    log_dir = train(config)
    # test(config, load_dir=log_dir)

    # worker = CentralizedEpisodicWorker(env, n_env, agent,
    #                                    max_episodes=100000,
    #                                    training=True,
    #                                    logger=logger,
    #                                    log_interval=config.log_interval,
    #                                    max_steps_per_ep=10000,
    #                                    render=True,
    #                                    )
    # worker.run()
