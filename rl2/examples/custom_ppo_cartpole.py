import gym
# from rl2.logging import Logger
import torch
from torch import nn

from rl2.agents.ppo import PPOAgent
from rl2.models.torch.actor_critic import ActorCriticModel
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

env = gym.make('Breakout-v4')
input_shape = env.observation_space.shape
if len(input_shape) > 1:
    input_shape = (input_shape[-1], *input_shape[:-1])
# print(input_shape)
encoder = nn.Conv2d(input_shape).to(device)
actor = nn.Linear(encoder.out_shape, env.action_space.n).to(device)
critic = nn.Linear(encoder.out_shape, 1).to(device)

model = ActorCriticModel(input_shape=input_shape, encoder=encoder, actor=actor, critic=critic)

config = {
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'clip_param': 0.9,
    'lr': 2.5e-4,
    'batch_size': 512,
    'num_epochs': 4,
}

agent = PPOAgent(model, config=config)
worker = MaxStepWorker(env=env, agent=agent, max_steps=1e7)

worker.run()

torch.save(agent.model.state_dict())
##########################################
model = ActorCriticModel(input_shape=input_shape, encoder=encoder, actor=actor, critic=critic)
torch.load(agent.model.state_dict())
