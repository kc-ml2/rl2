from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict
import collections
from collections.abc import Iterable
from torch.distributions import Distribution

from rl2.agents.configs import DEFAULT_MADDPG_CONFIG
from rl2.agents.base import Agent, MAgent
from rl2.agents.ddpg import DDPGModel
from rl2.models.torch.base import TorchModel
from rl2.agents.ddpg import DDPGModel
from rl2.buffers.base import ReplayBuffer, ExperienceReplay

import time


def loss_func_ac(transitions,
              models: TorchModel,
              **kwargs) -> List[torch.tensor]:
    index = kwargs['index']
    gamma = kwargs['gamma']
    mus = []
    s = None
    for data, model in zip(transitions, models):
        obs, action, reward, done, obs_ = tuple(
            map(lambda x: torch.from_numpy(x).float().to(model.device), data))
        if model.enc_ac is not None:
            state_ac = model.enc_ac(obs)
            state_cr = model.enc_cr(obs)
        else:
            state_ac = obs
            state_cr = obs
        if model.index != index:
            with torch.no_grad():
                mu = model.mu(state_ac)
                if model.discrete:
                    mu = F.gumbel_softmax(mu, tau=1, hard=True, dim=-1)
                    # mu = F.softmax(mu, dim=-1)
        else:
            mu = model.mu(state_ac)
            if model.discrete:
                mu = F.gumbel_softmax(mu, tau=1, hard=True, dim=-1)
            s = state_cr
        mus.append(mu)

    mus = torch.cat(mus, dim=-1)
    q_ac = models[index].q(torch.cat([s, mus], dim=-1))

    l_ac = -q_ac.mean()

    return l_ac

def loss_func_cr(transitions,
              models: TorchModel,
              **kwargs) -> List[torch.tensor]:
    index = kwargs['index']
    gamma = kwargs['gamma']
    mu_trgs, acs = [], []
    s = None
    for data, model in zip(transitions, models):
        obs, action, reward, done, obs_ = tuple(
            map(lambda x: torch.from_numpy(x).float().to(model.device), data))
        if model.enc_ac is not None:
            state_ac = model.enc_ac(obs)
            state_cr = model.enc_cr(obs)
            state_ac_ = model.enc_ac_trg(obs_)
            state_cr_ = model.enc_cr_trg(obs_)
        else:
            state_ac = state_cr = obs
            state_ac_ = state_cr_ = obs_
        if model.index == index:
            s = state_cr
            r = reward
            d = done
        with torch.no_grad():
            mu_trg = model.mu_trg(state_ac_)
            mu_trg = F.gumbel_softmax(mu_trg, tau=1, hard=True, dim=-1)
        mu_trgs.append(mu_trg)
        acs.append(action)

    mu_trgs = torch.cat(mu_trgs, dim=-1)
    acs = torch.cat(acs, dim=-1)

    with torch.no_grad():
        q_trg = models[index].q_trg(torch.cat([state_cr_, mu_trgs], dim=-1))
        bellman_trg = r + gamma * (1 - d) * q_trg
    q_cr = models[index].q(torch.cat([s, acs], dim=-1))

    # l_cr = F.smooth_l1_loss(q_cr, bellman_trg)
    l_cr = F.mse_loss(q_cr, bellman_trg)

    return l_cr


class MADDPGModel(DDPGModel):
    def __init__(self,
                 observation_shape,
                 action_shape,
                 joint_action_shape,
                 index,
                 actor: torch.nn.Module = None,
                 critic: torch.nn.Module = None,
                 encoder: torch.nn.Module = None,
                 encoder_dim : int = None,
                 optim_ac: str = None,
                 optim_cr: str = None,
                 discrete: bool = False,
                 flatten: bool = False,
                 reorder: bool = False,
                 **kwargs):

        super().__init__(observation_shape, action_shape,
                         encoder=encoder, encoder_dim=encoder_dim,
                         discrete=discrete, flatten=flatten, reorder=reorder,
                         config=DEFAULT_MADDPG_CONFIG, **kwargs)

        obs_dim = observation_shape[0]
        if len(observation_shape) == 3:
            obs_dim = encoder_dim
        if optim_cr is None:
            optim_cr = "torch.optim.Adam"
        self.q, self.optim_cr, self.q_trg = self._make_mlp_optim_target(
            critic,
            obs_dim + joint_action_shape[0],
            1,
            optim_cr
        )
        self.q = self.q.to(self.device)
        self.q_trg = self.q_trg.to(self.device)
        self.index = index

    def forward(self, obs, joint_ac: Iterable) -> Distribution:
        obs = obs.to(self.device)
        state_ac = self.enc_ac(obs)
        state_cr = self.enc_cr(obs)
        ac_dist = self.mu(state_ac)
        # act = ac_dist.mean
        joint_ac[self.index] = ac_dist
        joint_act = torch.cat(joint_ac, dim=-1)
        val_dist = self.q(state_cr, joint_act)

        return ac_dist, val_dist


class MADDPGAgent(MAgent):
    def __init__(self,
                 models: List[MADDPGModel],
                 update_interval=1,
                 train_interval=1,
                 num_epochs=1,
                 # FIXME: handle device in model class
                 device=None,
                 buffer_cls=ExperienceReplay,
                 buffer_kwargs=None,
                 **kwargs):

        self.config = EasyDict(DEFAULT_MADDPG_CONFIG)
        if 'config' in kwargs.keys():
            self.config = EasyDict(kwargs['config'])
        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.config.buffer_size}

        super().__init__(models, update_interval, num_epochs,
                         buffer_cls,
                         buffer_kwargs)

        self.loss_func_ac = loss_func_ac
        self.loss_func_cr = loss_func_cr

        # TODO: change these to get form kwargs
        start_eps = 0.9
        end_eps = 0.01
        self.eps = start_eps
        self.eps_func = lambda x, y: max(end_eps, x - start_eps / y)
        self.explore_steps = 5e5
        self.curr_step = 0.0

        # Temporary logging variables
        self.log_step = self.config.get('log_interval', 1000)
        self.mean_rew = collections.deque(maxlen=100)
        self.loss_cr = 0.0
        self.loss_ac = 0.0
        self.epi_score = 0.0

    def act(self, obss: List[np.ndarray]) -> List[np.ndarray]:
        self.eps = self.eps_func(self.eps, self.explore_steps)
        actions = []
        for model, obs in zip(self.models, obss):
            if len(obs.shape) in (1, 3):
                obs = np.expand_dims(obs, axis=0)
            action = model.act(obs)
            if model.discrete:
                if np.random.random() < self.eps:
                    _action = []
                    for ac in action:
                        num_actions = model.action_shape[0]
                        _action.append(
                            np.random.randint(num_actions))
                    action = np.array(_action).item()
                else:
                    action = np.argmax(action, axis=-1).item()
            else:
                action += self.eps * np.random.randn(*action.shape)
            actions.append(action)

        return actions

    def step(self, s, a, r, d, s_):
        if self.models[0].discrete:
            a_onehot = []
            for i, _a in enumerate(a):
                a_onehot.append(np.eye(self.models[i].action_shape[0])[_a])
            a = a_onehot
        self.collect(s, a, r, d, s_)
        self.epi_score += sum(r)
        if all(d):
            self.mean_rew.append(self.epi_score)
            self.epi_score = 0.0

        start_train = self.curr_step > self.init_collect
        if self.curr_step % self.train_interval == 0 and start_train:
            for _ in range(self.config.num_epochs):
                self.train()
        if self.curr_step % self.update_interval == 0 and start_train:
            for model in self.models:
                model.update_trg()

        if self.curr_step % self.log_step == 0 and start_train:
            curr_time = time.strftime('[%Y-%m-%d %I:%M:%S %p]', time.localtime())
            print(curr_time,
                  self.curr_step, sum(list(self.mean_rew)) / len(list(self.mean_rew)),
                  self.loss_ac, self.loss_cr)
        self.curr_step += 1

    def train(self):
        losses = []
        for i, model in enumerate(self.models):
            # TODO: get the buffer size
            sample_idx = np.random.randint(self.buffers[i].curr_size,
                                           size=self.batch_size)
            transitions = [buffer.sample(self.batch_size, idx=sample_idx)
                           for buffer in self.buffers]
            # transitions.insert(i, main_transition)
            loss_cr = self.loss_func_cr(transitions, self.models, index=i,
                                        gamma=self.gamma)
            model.step_cr(loss_cr)
            self.loss_cr = loss_cr.item()
            loss_ac = self.loss_func_ac(transitions, self.models, index=i,
                                        gamma=self.gamma)
            model.step_ac(loss_ac)
            self.loss_ac = loss_ac.item()

    def collect(self, obss: Iterable, acs: Iterable, rews: Iterable,
                dones: Iterable, obss_p: Iterable):
        # Store given observations
        for i, (obs, ac, rew, done, obs_p) in enumerate(
                zip(obss, acs, rews, dones, obss_p)):
            self.buffers[i].push(obs, ac, rew, done, obs_p)


if __name__ == '__main__':
    m = MADDPGModel()
    c = MADDPGAgent()
