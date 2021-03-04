import gym
import marlenv
# from rl2.logging import Logger
import torch
from torch import nn

from rl2.agents.maddpg import MADDPGAgent, MADDPGModel
from rl2.agents.configs import DEFAULT_MADDPG_CONFIG
from rl2.workers.multi_agent import SelfMaxStepWorker


"""
you might want to modify
1. layer architecture -> just pass nn.Module to predefined models
2. which distributions to use -> implement model from interfaces e.g. implement ActorCritic for custom PPO
3. how to sample distributions -> customize Agent
etc...

below example just changes 1. and some hparams
"""

custom_rew = {
    'fruit': 10.0,
    'kill': 0.0,
    'lose': -10.0,
    'win': 0.0,
    'time':0.0,
}
env = gym.make('Snake-v1', height=10, width=10, reward_dict=custom_rew)

config = DEFAULT_MADDPG_CONFIG

myconfig = {
    # TODO: Example config
}

class Encoder(nn.Module):
    def __init__(self, obs_shape, out_shape):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        dummy = torch.zeros(1, obs_shape[-1], *obs_shape[:-1])
        with torch.no_grad():
            ir_shape = self.body(dummy).flatten().shape[-1]
        self.head = nn.Linear(ir_shape, out_shape)

    def forward(self, x):
        ir = self.body(x)
        ir = ir.flatten(start_dim=1)
        ir = self.head(ir)
        return ir


if __name__ == '__main__':
    observation_shape = env.observation_space.shape
    assert hasattr(env.action_space, 'n')
    joint_action_shape = (sum(env.action_space.n),)
    action_shape = [(ac_dim,) for ac_dim in env.action_space.n]
    # action_shape = (env.action_space.n,) if hasattr(
    #     env.action_space, 'n') else env.action_space.shape
    models = []
    for i, (obs_shape, ac_shape) in enumerate(zip(observation_shape, action_shape)):
        model = MADDPGModel(obs_shape, ac_shape, joint_action_shape, i,
                            encoder=Encoder(obs_shape, 64), encoder_dim=64,
                            reorder=True, discrete=True, device='cuda')
        models.append(model)
    agent = MADDPGAgent(models, config=config)
    observation_shape = env.observation_space.shape
    worker = SelfMaxStepWorker(env=env, agent=agent, training=True, max_steps=1e7)
    # worker = EpisodicWorker(env=env,
    # agent=agent,
    # training=True,
    # max_episodes=10,
    # max_steps_per_ep=1e4,
    # max_steps=1e7,
    # render=True)

    worker.run()
