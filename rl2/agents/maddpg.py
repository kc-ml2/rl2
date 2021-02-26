from typing import List
import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Iterable
from torch.distributions import Distribution

from rl2.agents.base import Agent
from rl2.models.torch.base import TorchModel
from rl2.models.torch.ddpg import DDPGModel
from rl2.buffers import ReplayBuffer
from rl2.networks import MLP


def loss_func(transitions,
              models: TorchModel,
              **kwargs) -> List[torch.tensor]:
    index = kwargs['index']
    gamma = kwargs['gamma']
    mus, mu_trgs, acs  = [], [], []
    s, a = None, None
    for data, model in zip(transitions, models):
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

        if actor is None and len(observation_shape) == 1:
            actor = MLP(in_shape=observation_shape[0],
                        out_shape=action_shape[0])

        if critic is None:
            # FIXME: Acition_dim management
            critic = MLP(
                in_shape=(observation_shape[0] + joint_action_shape[0]),
                out_shape=1)

        self.mu = actor
        self.q = critic

        if optim_ac is None:
            self.optim_ac = "torch.optim.Adam"
        if optim_cr is None:
            self.optim_cr = "torch.optim.Adam"

        self.optim_ac = self.get_optimizer_by_name(
            modules=[self.mu], optim_name=self.optim_ac)
        self.optim_cr = self.get_optimizer_by_name(
            modules=[self.q], optim_name=self.optim_cr)

        self.mu_trg = copy.deepcopy(self.mu)
        self.q_trg = copy.deepcopy(self.q)

        for p_mu, p_q in zip(self.mu_trg.parameters(),
                             self.q_trg.parameters()):
            p_mu.requires_grad = False
            p_q.requires_grad = False

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
                 buffer_cls=ReplayBuffer,
                 buffer_kwargs=None,
                 **kwargs):

        # config = kwargs['config']

        if buffer_kwargs is None:
            buffer_kwargs = {'size': 10}

        super().__init__(**kwargs)

        self.models = []
        self.loss_func = loss_func
        full_ac_shape = np.asarray(ac_shapes).sum()

        # TODO: change these to get form kwargs
        start_eps = 0.9
        end_eps = 0.01
        self.eps = start_eps
        self.eps_func = lambda x, y: max(end_eps, x - start_eps / y)
        self.explore_steps = 1e5

        self.batch_size = kwargs.get('batch_size', 32)
        self.buffers = [ReplayBuffer(**buffer_kwargs) for model in self.models]

    def act(self, obss: List[np.array]) -> List[np.array]:
        noise_scale = 0.1  # TODO: change this to attribute
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
