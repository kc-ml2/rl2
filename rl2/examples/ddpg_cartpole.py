import gym
# from rl2.logging import Logger
import torch
from torch import nn

from rl2.agents.ddpg import DDPGAgent, DDPGModel
from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.workers.base import EpisodicWorker, MaxStepWorker


"""
you might want to modify 
1. layer architecture -> just pass nn.Module to predefined models
2. which distributions to use -> implement model from interfaces e.g. implement ActorCritic for custom PPO
3. how to sample distributions -> customize Agent
etc...

below example just changes 1. and some hparams
"""

env = gym.make('MountainCarContinuous-v0')
config = DEFAULT_DDPG_CONFIG

myconfig = {
    # TODO: Example config
}

if __name__ == '__main__':
    observation_shape = env.observation_space.shape
    action_shape = (env.action_space.n,) if hasattr(
        env.action_space, 'n') else env.action_space.shape

    model = DDPGModel(observation_shape=observation_shape,
                      action_shape=action_shape)
    agent = DDPGAgent(model)
    # worker = MaxStepWorker(env=env, agent=agent, training=True, max_steps=1e7)
    worker = EpisodicWorker(env=env,
                            agent=agent,
                            training=True,
                            max_episodes=10,
                            max_steps_per_ep=1e4,
                            max_steps=1e7,
                            render=True)

    worker.run()
