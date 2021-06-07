import os
import json
from easydict import EasyDict

import marlenv
from marlenv.wrappers import make_snake

from rl2.examples.temp_logger import Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers.multi_agent import MAMaxStepWorker, MAEpisodicWorker

# FIXME: Remove later
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
                     n_env=props.n_env,
                     batch_size=config.batch_size,
                     num_epochs=config.epoch,
                     buffer_kwargs={'size': config.train_interval,
                                    'n_env': props.n_env})
    return agent


def train(config):
    logger = Logger(name='MATRAIN', args=config)
    env, observation_shape, action_shape, props = make_snake(
        n_env=config.n_env,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack
    )

    agents = []
    for _ in range(config.num_snakes):
        agents.append(ppo(observation_shape, action_shape, config, props))

    worker = MAMaxStepWorker(env, props.n_env, agents,
                             max_steps=int(1e5),
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
        model_dir = os.path.join(load_dir, "ckpt")
        with open(config_file, "r") as config_f:
            _config = EasyDict(json.load(config_f))
    else:
        model_dir = None
    logger = Logger(name='MATEST', args=config)

    env, observation_shape, action_shape, props = make_snake(
        n_env=1,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack
    )
    agents = []
    for i in range(config.num_snakes):
        if model_dir is not None:
            model_file = os.path.join(model_dir,
                                      f'agent{i}', '100k', 'PPOModel.pt')
            # model_file = os.path.join(model_dir,
            #                           '100k', 'PPOModel.pt')
        else:
            model_file = None
        agents.append(
            ppo(observation_shape, action_shape, config, props,
                load_dir=model_file)
        )

    worker = MAEpisodicWorker(env, props.n_env, agents,
                              max_episodes=3, training=False,
                              render=True,
                              render_interval=1,
                              logger=logger)
    worker.run()


if __name__ == "__main__":
    myconfig = {
        'n_env': 64,
        'num_snakes': 4,
        'width': 20,
        'height': 20,
        'vision_range': 5,
        'frame_stack': 2,
        'batch_size': 512,
        'epoch': 4,
        'train_interval': 128,
        'log_level': 10,
        'log_interval': 50000,
        'save_interval': 100000,
        'lr': 1e-4,
        'gamma': 0.99,
        'grad_clip': 10,
        'tag': 'DEBUG',
    }
    config = EasyDict(myconfig)

    log_dir = train(config)
    # log_dir = 'runs/DEBUG/20210505180137'
    #test(config, load_dir=log_dir)
