import gym
# from rl2.logging import Logger
import torch
from torch import nn

from rl2.agents.ddpg import DDPGAgent, DDPGModel
from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.workers.base import MaxStepWorker


"""
you might want to modify 
1. layer architecture -> just pass nn.Module to predefined models
2. which distributions to use -> implement model from interfaces e.g. implement ActorCritic for custom PPO
3. how to sample distributions -> customize Agent
etc...

below example just changes 1. and some hparams
"""

device = 'cpu'

env = gym.make('CartPole-v1')
input_shape = env.observation_space.shape
if len(input_shape) > 1:
    input_shape = (input_shape[-1], *input_shape[:-1])

config = DEFAULT_DDPG_CONFIG

myconfig = {
    # TODO
}
print(input_shape)

if __name__ == '__main__':
    model = DDPGModel(input_shape=input_shape, enc_dim=128)
    agent = DDPGAgent(model, config=config)
    worker = MaxStepWorker(env=env, agent=agent, max_steps=1e7)

    worker.run()
