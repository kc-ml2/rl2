from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
from collections.abc import Iterable
from torch.distributions import Distribution

from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.agents.base import Agent
from rl2.models.torch.base import TorchModel
from rl2.models.torch.ddpg import DDPGModel
from rl2.buffers import ReplayBuffer, ExperienceReplay


def loss_func(transitions,
              models: TorchModel,
              **kwargs) -> List[torch.tensor]:
    index = kwargs['index']
    gamma = kwargs['gamma']
    mus, mu_trgs, acs = [], [], []
    s = None
    for data, model in zip(transitions, models):
        data = list(data)
        state, action, reward, done, state_ = tuple(
            map(lambda x: torch.from_numpy(x).float().to(models), data))
        if model.index != index:
            with torch.no_grad():
                mu = model.mu(state)
                mu_trg = model.mu_trg(state_)
        else:
            mu = model.mu(state)
            mu_trg = model.mu_trg(state_)
            s = state
        mus.append(mu)
        mu_trgs.append(mu_trg)
        acs.append(action)

    mus = torch.cat(mus, dim=-1)
    mu_trgs = torch.cat(mu_trgs, dim=-1)
    acs = torch.cat(acs, dim=-1)

    q_trg = models[index].q_trg(torch.cat([state_, mu_trgs], dim=-1))
    bellman_trg = reward + gamma * (1 - done) * q_trg
    q_ac = models[index].q(torch.cat([s, mus], dim=-1))
    q_cr = models[index].q(torch.cat([s, acs], dim=-1))

    l_ac = -q_ac.mean()
    l_cr = F.smooth_l1_loss(q_cr, bellman_trg)

    loss = [l_ac, l_cr]

    return loss


class MADDPGModel(DDPGModel):
    def __init__(self,
                 observation_shape,
                 action_shape,
                 joint_action_shape,
                 index,
                 actor: torch.nn.Module = None,
                 critic: torch.nn.Module = None,
                 optim_ac: str = None,
                 optim_cr: str = None,
                 **kwargs):

        super().__init__(observation_shape, action_shape, **kwargs)

        self.config = EasyDict(DEFAULT_DDPG_CONFIG)
        if 'config' in kwargs.keys():
            self.config = EasyDict(kwargs['config'])
        if optim_ac is None:
            optim_ac = "torch.optim.Adam"
        if optim_cr is None:
            optim_cr = "torch.optim.Adam"

        self.mu, self.optim_ac, self.mu_trg = self._make_mlp_optim_target(
            actor,
            observation_shape[0],
            action_shape[0],
            optim_ac
        )
        self.q, self.optim_cr, self.q_trg = self._make_mlp_optim_target(
            critic,
            observation_shape[0] + joint_action_shape[0],
            1,
            optim_cr
        )
        self.index = index

    def forward(self, obs, joint_ac: Iterable) -> Distribution:
        ac_dist = self.mu(obs)
        act = ac_dist.mean
        joint_ac[self.index] = act
        joint_act = torch.cat(joint_ac, dim=-1)
        val_dist = self.q(obs, joint_act)

        return ac_dist, val_dist


class MADDPGAgent(Agent):
    def __init__(self,
                 models: List[DDPGModel],
                 update_interval=1,
                 num_epochs=1,
                 # FIXME: handle device in model class
                 device=None,
                 buffer_cls=ExperienceReplay,
                 buffer_kwargs=None,
                 **kwargs):

        # config = kwargs['config']

        if buffer_kwargs is None:
            buffer_kwargs = {'size': 10}

        super().__init__(**kwargs)

        self.models = []
        self.loss_func = loss_func

        # TODO: change these to get form kwargs
        start_eps = 0.9
        end_eps = 0.01
        self.eps = start_eps
        self.eps_func = lambda x, y: max(end_eps, x - start_eps / y)
        self.explore_steps = 1e5

        self.batch_size = kwargs.get('batch_size', 32)
        self.buffers = [ReplayBuffer(**buffer_kwargs) for model in self.models]

    def act(self, obss: List[np.array]) -> List[np.array]:
        self.eps = self.eps_func(self.eps, self.explore_steps)
        actions = []
        for model, obs in zip(self.models, obss):
            action = model.act(obs)
            noise = self.eps * np.random.randn_like(action)
            actions.append(action + noise)

        return actions

    def train(self):
        losses = []
        for i, model in enumerate(self.models):
            # TODO: get the buffer size
            main_transition = self.buffer[i].sample(self.batch_size,
                                                    return_idx=True)
            transitions = [buffer[i].sample(self.batch_size,
                                            idx=main_transition.idx)
                           for buffer in self.buffer]
            transitions.insert(i, main_transition)
            loss = self.loss_func(transitions, self.models)
        for model, loss in zip(self.models, losses):
            model.step(loss)

    def collect(self, obss: Iterable, acs: Iterable, rews: Iterable,
                dones: Iterable, obss_p: Iterable):
        # Store given observations
        for i, (obs, ac, rew, done, obs_p) in enumerate(
            zip(obss, acs, rews, dones, obss_p)):
            self.buffers[i].push(obs, ac, rew, done, obs_p)
