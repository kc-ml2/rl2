import json
import os

from marlenv.wrappers import make_snake

from rl2.agents.configs import DEFAULT_DQN_CONFIG
from rl2.agents.dqn import DQNAgent, DQNModel
from rl2.examples.temp_logger import Logger
from rl2.utils import EasyDict
from rl2.workers.base import EpisodicWorker, MaxStepWorker


def dqn(obs_shape, ac_shape, config, props, load_dir=None):
    model = DQNModel(
        observation_shape=obs_shape,
        action_shape=ac_shape,
        double=config.double,
        recurrent=config.recurrent,
        optimizer=config.optimizer,
        lr=config.lr,
        grad_clip=config.grad_clip,
        polyak=config.polyak,
        reorder=True,
        discrete=props['discrete'],
        high=props['high']
    )
    if load_dir is not None:
        model.load(load_dir)
    agent = DQNAgent(
        model,
        update_interval=config.update_interval,
        train_interval=config.train_interval,
        num_epochs=config.num_epochs,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        decay_step=config.decay_step,
        eps=config.eps,
        gamma=config.gamma,
        log_interval=config.log_interval,
    )
    return agent


def train(config):
    logger = Logger(name='DEFAULT', args=config)
    custom_reward = {
        'fruit': 1.0,
        'kill': 0.0,
        'lose': 0.0,
        'win': 0.0,
        'time': 0.0
    }
    env, observation_shape, action_shape, props = make_snake(
        num_envs=1,
        num_snakes=1,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack,
        reward_dict=custom_reward,
    )
    agent = dqn(observation_shape, action_shape, config, props)
    worker = MaxStepWorker(env, agent,
                           num_envs=props['num_envs'],
                           max_steps=config.max_step, training=True,
                           log_interval=config.log_interval,
                           render=True,
                           render_mode='rgb_array',
                           render_interval=100000,
                           is_save=True,
                           save_interval=config.save_interval,
                           logger=logger)

    worker.run()
    return logger.log_dir


def test(config, load_dir=None):
    # Test phase
    if log_dir is not None:
        config_file = os.path.join(load_dir, "config.json")
        model_file = os.path.join(load_dir, "ckpt", "1k", "DQNModel.pt")
        with open(config_file, "r") as config_f:
            config = EasyDict(json.load(config_f))
    logger = Logger(name='TUTORIAL', args=config)

    env, observation_shape, action_shape, props = make_snake(
        num_envs=1,
        num_snakes=1,
        width=config.width,
        height=config.height,
        vision_range=config.vision_range,
        frame_stack=config.frame_stack
    )
    agent = dqn(observation_shape, action_shape, config, props,
                load_dir=model_file)
    worker = EpisodicWorker(env=env,
                            num_envs=1,
                            agent=agent,
                            training=False,
                            max_episodes=3,
                            max_steps_per_ep=1e4,
                            log_interval=config.log_interval,
                            render=True,
                            logger=logger,
                            is_save=False,
                            render_mode='rgb_array',
                            render_interval=1,
                            )
    worker.run()


if __name__ == '__main__':
    # Use Default config
    config = DEFAULT_DQN_CONFIG

    # Or Customize your config
    myconfig = {
        'width': 20,
        'height': 20,
        'vision_range': 5,
        'frame_stack': 2,
        'buffer_size': int(1e5),
        'batch_size': 32,
        'num_epochs': 1,
        'max_step': int(5e5),
        'update_interval': int(1e4),
        'train_interval': 1,
        'log_interval': 20000,
        'save_interval': int(1e3),
        'optimizer': 'torch.optim.Adam',
        'lr': 1e-3,
        'recurrent': True,
        'gamma': 0.99,
        'eps': 0.001,
        'polyak': 0,
        'decay_step': int(1e5),
        'grad_clip': 10,
        'tag': 'DDQN/SNAKE',
        'double': True,
        'log_level': 10,
    }
    config = EasyDict(myconfig)

    log_dir = train(config)
    test(config, load_dir=log_dir)
