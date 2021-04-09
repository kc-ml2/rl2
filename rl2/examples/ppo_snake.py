import os
import json
from easydict import EasyDict

from marlenv.wrappers import make_snake

from rl2.examples.temp_logger import Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers import MaxStepWorker, EpisodicWorker


def ppo(obs_shape, ac_shape, config, props, load_dir=None):
    model = PPOModel(obs_shape,
                     ac_shape,
                     recurrent=config.recurrent,
                     discrete=True,
                     reorder=props.reorder,
                     optimizer=config.optimizer,
                     high=props.high)
    agent = PPOAgent(model,
                     train_interval=config.train_interval,
                     n_env=props.n_env,
                     batch_size=config.batch_size,
                     num_epochs=config.epoch,
                     buffer_kwargs={'size': config.train_interval,
                                    'n_env': props.n_env})
    return agent


def train(config):
    # Train phase
    logger = Logger(name='TUTORIAL', args=config)
    custom_reward = {
        'fruit': 1.0,
        'kill': 0.0,
        'lose': 0.0,
        'win': 0.0,
        'time': 0.1
    }
    env, observation_shape, action_shape, props = make_snake(
        n_env=config.n_env,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_rang=config.vision_range,
        frame_stack=config.frame_stack,
        reward_dict=custom_reward,
    )
    agent = ppo(observation_shape, action_shape, config, props)
    worker = MaxStepWorker(env, props.n_env, agent,
                           max_steps=config.max_step, training=True,
                           log_interval=config.log_interval,
                           render=True,
                           render_interval=200000,
                           is_save=True,
                           save_interval=config.save_interval,
                           logger=logger)
    worker.run()
    return logger.log_dir


def test(config, load_dir=None):
    # Test phase
    if log_dir is not None:
        config_file = os.path.join(load_dir, "config.json")
        model_file = os.path.join(load_dir, "ckpt", "10k.pt")
        with open(config_file, "r") as config_f:
            config = EasyDict(json.load(config_f))
    logger = Logger(name='TUTORIAL', args=config)

    env, observation_shape, action_shape, props = make_snake(
        n_env=1,
        num_snakes=config.num_snakes,
        width=config.width,
        height=config.height,
        vision_rang=config.vision_range,
        frame_stack=config.frame_stack
    )
    agent = ppo(observation_shape, action_shape, config, props,
                load_dir=model_file)
    worker = EpisodicWorker(env, props.n_env, agent,
                            max_episodes=3, training=False,
                            render=True,
                            render_interval=1,
                            logger=logger)
    worker.run()


if __name__ == "__main__":
    # This can be replaced with argparser, click, etc.
    myconfig = {
        'n_env': 2,
        'num_snakes': 1,
        'width': 20,
        'height': 20,
        'vision_range': 5,
        'frame_stack': 2,
        'train_interval': 128,
        'epoch': 4,
        'batch_size': 128,
        'max_step': int(2e4),
        'optimizer': 'torch.optim.RMSprop',
        'recurrent': False,
        'log_interval': 20000,
        'log_level': 10,
        'save_interval': 10000,
        'tag': 'DEBUG/',
    }
    config = EasyDict(myconfig)

    log_dir = train(config)
    test(config, load_dir=log_dir)
