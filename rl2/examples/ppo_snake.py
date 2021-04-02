from easydict import EasyDict

from marlenv.wrappers import make_snake

from rl2.examples.temp_logger import Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers import MaxStepWorker


myconfig = {
    'n_env': 64,
    'num_snakes': 1,
    'width': 20,
    'height': 20,
    'vision_range': 5,
    'frame_stack': 2,
    'train_interval': 128,
    'epoch': 4,
    'batch_size': 512,
    'max_step': int(5e6),
    'log_interval': 20000,
    'log_level': 10,
    'tag': 'DEBUG/',
}
config = EasyDict(myconfig)


def ppo(config, props):
    model = PPOModel(observation_shape,
                     action_shape,
                     recurrent=True,
                     discrete=True,
                     reorder=props.reorder,
                     optimizer='torch.optim.RMSprop',
                     high=props.high)
    train_interval = config.train_interval
    epoch = config.epoch
    batch_size = config.batch_size
    agent = PPOAgent(model,
                     train_interval=train_interval,
                     n_env=props.n_env,
                     batch_size=batch_size,
                     num_epochs=epoch,
                     buffer_kwargs={'size': train_interval,
                                    'n_env': props.n_env})
    return agent


if __name__ == "__main__":
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
        frame_stack=config.frame_stack
    )

    agent = ppo(config, props)
    worker = MaxStepWorker(env, props.n_env, agent,
                           max_steps=config.max_step, training=True,
                           log_interval=config.log_interval,
                           render=True,
                           render_interval=200000,
                           is_save=True,
                           logger=logger)

    worker.run()
