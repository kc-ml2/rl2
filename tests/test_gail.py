# produce expert data
import pickle

from rl2.agents.ppo import PPOModel, PPOAgent
from marlenv.wrappers import make_snake
from rl2.examples.temp_logger import Logger
from rl2.models.base import BranchModel
from rl2.utils import EasyDict
from rl2.workers import EpisodicWorker, MaxStepWorker

custom_reward = {
    'fruit': 1.0,
    'kill': 0.0,
    'lose': 0.0,
    'win': 0.0,
    'time': 0.0
}
config = {
    'num_envs': 1,
    'num_snakes': 1,
    'width': 7,
    'height': 7,
    'vision_range': 5,
    'frame_stack': 2,
    'train_interval': 0,
    'epoch': 4,
    'batch_size': 512,
    'max_step': 1200000,
    'optimizer': 'torch.optim.RMSprop',
    'recurrent': False,
    'log_interval': 20000,
    'log_level': 10,
    'save_interval': 1000000,
    'tag': 'EXPERT_DATA',
}
config = EasyDict(config)

env, observation_shape, action_shape, props = make_snake(
    num_envs=config.num_envs,
    num_snakes=config.num_snakes,
    width=config.width,
    height=config.height,
    vision_range=config.vision_range,
    frame_stack=config.frame_stack,
    reward_dict=custom_reward,
)
# print(props)
model = PPOModel(
    observation_shape,
    action_shape,
    recurrent=config.recurrent,
    discrete=True,
    reorder=props['reorder'],
    optimizer=config.optimizer,
    high=props['high']
)
saved_model_dir = 'PPOModel.pt'
model.load(saved_model_dir)
print(config.train_interval)
agent = PPOAgent(
    model,
    train_interval=config.train_interval,
    # num_envs=props.num_envs,
    # batch_size=config.batch_size,
    # num_epochs=config.epoch,
    # buffer_kwargs={
    #     'size': config.train_interval,
    #     'num_envs': props.num_envs
    # }
)
erange = (10, 31)
worker = EpisodicWorker(
    env, num_envs=config.num_envs,
    agent=agent,
    max_episodes=1024,
    save_erange=erange
)
with worker:
    worker.run()

# save expert data
with open('expert_data.pickle', 'wb') as fp:
    pickle.dump(worker.trajectories, fp)

# load expert data
with open('expert_data.pickle', 'rb') as fp:
    expert_data = pickle.load(fp)
# gail
config = {
    'num_envs': 64,
    'num_snakes': 1,
    'width': 7,
    'height': 7,
    'vision_range': 5,
    'frame_stack': 2,
    'train_interval': 128,
    'epoch': 4,
    'batch_size': 512,
    'max_step': 1200000,
    'optimizer': 'torch.optim.RMSprop',
    'recurrent': False,
    'log_interval': 20000,
    'log_level': 10,
    'save_interval': 1000000,
    'tag': 'GAIL',
}
config = EasyDict(config)

logger = Logger(name='GAIL', args=config)
custom_reward = {
    'fruit': 1.0,
    'kill': 0.0,
    'lose': 0.0,
    'win': 0.0,
    'time': 0.0
}

env, observation_shape, action_shape, props = make_snake(
    num_envs=config.num_envs,
    num_snakes=config.num_snakes,
    width=config.width,
    height=config.height,
    vision_range=config.vision_range,
    frame_stack=config.frame_stack,
    reward_dict=custom_reward,
)

model = PPOModel(
    observation_shape,
    action_shape,
    recurrent=config.recurrent,
    discrete=True,
    reorder=props['reorder'],
    optimizer=config.optimizer,
    high=props['high']
)

agent = PPOAgent(
    model,
    train_interval=config.train_interval,
    num_envs=props['num_envs'],
    batch_size=config.batch_size,
    num_epochs=config.epoch,
    buffer_kwargs={
        'size': config.train_interval,
        'num_envs': props['num_envs']
    },
    use_gail=True,
    discriminator=BranchModel(
        (11 * 11 * 16 + 5,), (1,),
        discrete=False,
        deterministic=True,
        flatten=True
    ),
    expert_trajs=expert_data
)

worker = MaxStepWorker(
    env,
    agent,
    max_steps=config.max_step, training=True,
    log_interval=config.log_interval,
    render=True,
    render_interval=500000,
    is_save=True,
    save_interval=config.save_interval,
    logger=logger,
    num_envs=props['num_envs'],
)
with worker.as_saving():
    worker.run()
