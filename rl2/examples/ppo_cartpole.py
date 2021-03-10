import gym
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.workers import MaxStepWorker


env = gym.make("CartPole-v0")
observation_shape = env.observation_space.shape
action_shape = (env.action_space.n,)

model = PPOModel(observation_shape,
                 action_shape,
                 discrete=True)

train_interval = 512
num_env = 1
epoch = 4
batch_size = 128
agent = PPOAgent(model,
                 train_interval=train_interval,
                 batch_size=batch_size,
                 num_epochs=train_interval // batch_size * epoch,
                 buffer_kwargs={'size': train_interval * num_env})

worker = MaxStepWorker(env, agent, max_steps=int(1e6), training=True)

worker.run()

# TODO: vecenv, minibatch
