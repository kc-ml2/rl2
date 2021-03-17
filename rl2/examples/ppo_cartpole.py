import gym
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.async_vector_env import AsyncVectorEnv
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import marlenv
from marlenv.wrappers import SingleAgent
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.agents.ddpg import DDPGModel, DDPGAgent
from rl2.workers import MaxStepWorker


def make_cartpole(seed=None):
    def _make():
        env = gym.make("CartPole-v0")
        if seed is not None:
            env.seed(seed)
        return env
    return _make


def make_snake():
    def _make():
        env = gym.make("Snake-v1", num_snakes=1, width=10, height=10,
                       vision_range=5)
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

def make_vec():
    n_env = 64
    # dummyenv = gym.make("CartPole-v0")
    # env = AsyncVectorEnv([make_cartpole() for i in range(n_env)])
    dummyenv = make_snake()()
    env = AsyncVectorEnv([make_snake() for i in range(n_env)])
    observation_shape = dummyenv.observation_space.shape
    action_shape = (dummyenv.action_space.n,)
    high = dummyenv.observation_space.high
    return n_env, env, observation_shape, action_shape, high


n_env, env, observation_shape, action_shape, high = make_single()
reorder = True


def ppo():
    model = PPOModel(observation_shape,
                     action_shape,
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
                     num_epochs=(train_interval * n_env) // batch_size * epoch,
                     buffer_kwargs={'size': train_interval,
                                    'n_env': num_env})
    return agent


def ddpg():
    model = DDPGModel(observation_shape,
                      action_shape,
                      discrete=True,
                      reorder=reorder)
    train_interval = 1
    num_env = 1
    epoch = 1
    batch_size = 32
    agent = DDPGAgent(model,
                      train_interval=train_interval,
                      batch_size=batch_size,
                      num_epochs=epoch)
    return agent


# agent = ppo()
agent = ddpg()
worker = MaxStepWorker(env, n_env, agent, max_steps=int(5e6), training=True)

worker.run()

# TODO: vecenv, minibatch
