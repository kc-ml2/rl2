from easydict import EasyDict

import gym
# from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.async_vector_env import AsyncVectorEnv
from marlenv.wrappers import SingleMultiAgent, AsyncVectorMultiEnv

import marlenv
from marlenv.wrappers import SingleAgent

from rl2.examples.temp_logger import Logger
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers import MaxStepWorker


custom_reward = {
    'fruit': 1.0,
    'kill': 0.0,
    'lose': 0.0,
    'win': 0.0,
    'time': 0.1
}


def make_snake():
    def _make():
        env = gym.make("Snake-v1",
                       num_snakes=1, width=20, height=20,
                       vision_range=5, frame_stack=2,
                       reward_dict=custom_reward)
        env = SingleAgent(env)
        return env
    return _make


def make_single():
    n_env = 1
    env = make_snake()()
    observation_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)
    high = env.observation_space.high
    return n_env, env, observation_shape, action_shape, high


def make_vec(n_env=64):
    dummyenv = make_snake()()
    observation_shape = dummyenv.observation_space.shape
    action_shape = (dummyenv.action_space.n,)
    high = dummyenv.observation_space.high
    del dummyenv
    env = AsyncVectorMultiEnv([make_snake() for i in range(n_env)])
    return n_env, env, observation_shape, action_shape, high


n_env, env, observation_shape, action_shape, high = make_vec()
reorder = True


def ppo():
    model = PPOModel(observation_shape,
                     action_shape,
                     recurrent=True,
                     discrete=True,
                     reorder=reorder,
                     optimizer='torch.optim.RMSprop',
                     high=high)
    train_interval = 128
    num_env = n_env
    epoch = 4
    batch_size = 512
    agent = PPOAgent(model,
                     train_interval=train_interval,
                     n_env=n_env,
                     batch_size=batch_size,
                     num_epochs=epoch,
                     buffer_kwargs={'size': train_interval,
                                    'n_env': num_env})
    return agent


myconfig = {
    'tag': 'TUTORIAL_PPO/',
    'log_level': 10,
    'log_interval': 20000
}
config = EasyDict(myconfig)


if __name__ == "__main__":
    logger = Logger(name='TUTORIAL', args=config)
    agent = ppo()
    worker = MaxStepWorker(env, n_env, agent,
                           max_steps=int(5e6), training=True,
                           log_interval=config.log_interval,
                           render=True,
                           render_interval=200000,
                           is_save=True,
                           logger=logger)

    worker.run()
