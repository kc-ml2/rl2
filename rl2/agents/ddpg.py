import copy
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution
from rl2.agents.base import Agent
from rl2.buffers.base import ExperienceReplay, ReplayBuffer
from rl2.models.torch.base import PolicyBasedModel, ValueBasedModel
from rl2.networks.torch.networks import MLP


def loss_func_ac(data, model, **kwargs):
    s = torch.from_numpy(data[0]).float().to(model.device)
    if model.enc_ac is not None:
        s_ac = model.enc_ac(s)
        s_cr = model.enc_cr(s)
    else:
        s_ac = s_cr = s
    q_val = model.q(torch.cat([s_cr, torch.tanh(model.mu(s_ac))], dim=-1))
    loss = -q_val.mean()

    return loss


def loss_func_cr(data, model, **kwargs):
    s, a, r, d, s_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
    )
    if model.enc_cr is not None:
        s_cr = model.enc_cr(s)
        s_ac_ = model.enc_ac_trg(s_)
        s_cr_ = model.enc_cr_trg(s_)
    else:
        s_ac_ = s_cr_ = s_

    with torch.no_grad():
        a_trg = model.mu_trg(s_ac_)
        if model.discrete:
            a_trg = F.gumbel_softmax(a_trg, tau=1., hard=True)
        else:
            a_trg = torch.tanh(a_trg)
        v_trg = model.q_trg(torch.cat([s_cr_, a_trg], dim=-1))
        bellman_trg = r + kwargs['gamma'] * v_trg * (1-d)

    q = model.q(torch.cat([s_cr, a], dim=-1))
    loss = F.smooth_l1_loss(q, bellman_trg)

    return loss


class DummyEncoder(nn.Module):
    def __init__(self, net, reorder, flatten):
        super().__init__()
        self.net = net
        self.reorder = reorder
        self.flatten = flatten

    def forward(self, obs):
        if self.net is not None:
            if self.reorder:
                obs = obs.permute(0, 3, 1, 2)
            obs = self.net(obs)
        elif self.flatten:
            obs = obs.view(obs.shape[0], -1)
        return obs


class DDPGModel(PolicyBasedModel, ValueBasedModel):
    """
    predefined model
    (same one as original paper)
    """

    def __init__(self,
                 observation_shape,
                 action_shape,
                 actor: torch.nn.Module = None,
                 critic: torch.nn.Module = None,
                 encoder: torch.nn.Module = None,
                 encoder_dim: int = None,
                 optim_ac: str = 'torch.optim.Adam',
                 optim_cr: str = 'torch.optim.Adam',
                 lr_ac: float = 1e-4,
                 lr_cr: float = 1e-4,
                 grad_clip: float = 1e-2,
                 polyak: float = 0.995,
                 discrete: bool = False,
                 flatten: bool = False,
                 reorder: bool = False,
                 **kwargs):

        super().__init__(observation_shape, action_shape, **kwargs)

        if optim_ac is None:
            optim_ac = self.optim_ac
        if optim_cr is None:
            optim_cr = self.optim_cr

        self.grad_clip = grad_clip

        self.lr_ac = lr_ac
        self.lr_cr = lr_cr
        self.polyak = polyak
        self.is_save = kwargs['is_save']

        obs_dim = observation_shape[0]
        self.enc_ac = DummyEncoder(encoder, reorder, flatten).to(self.device)
        self.enc_cr = DummyEncoder(encoder, reorder, flatten).to(self.device)
        self.enc_ac_trg = copy.deepcopy(self.enc_ac)
        self.enc_cr_trg = copy.deepcopy(self.enc_cr)
        self.discrete = discrete
        if len(observation_shape) == 3:
            assert encoder is not None, 'Must provide an encoder for 2d input'
            assert encoder_dim is not None

            self.optim_enc_ac, self.enc_ac_trg = self._make_optim_target(
                self.enc_ac, optim_ac
            )
            self.optim_enc_cr, self.enc_cr_trg = self._make_optim_target(
                self.enc_cr, optim_cr
            )
            obs_dim = encoder_dim
            self.enc_ac, self.enc_cr, self.enc_ac_trg, self.enc_cr_trg = map(
                lambda x: x.to(self.device),
                self.enc_ac, self.enc_cr, self.enc_ac_trg, self.enc_cr_trg)
        self.mu, self.optim_ac, self.mu_trg = self._make_mlp_optim_target(
            actor, obs_dim, action_shape[0], optim_ac, lr=self.lr_ac
        )
        self.q, self.optim_cr, self.q_trg = self._make_mlp_optim_target(
            critic, obs_dim + action_shape[0], 1, optim_cr, lr=self.lr_cr
        )
        self.mu, self.mu_trg, self.q, self.q_trg = map(
            lambda x: x.to(self.device),
            [self.mu, self.mu_trg, self.q, self.q_trg]
        )
        self.mu, self.q, self.mu_trg, self.q_trg = map(
            lambda x: x.to(self.device),
            [self.mu, self.q, self.mu_trg, self.q_trg])

    def _make_mlp_optim_target(self, network,
                               num_input, num_output, optim_name, **kwargs):
        if network is None:
            network = MLP(in_shape=num_input, out_shape=num_output,
                          hidden=[128, 128])
        optimizer, target_network = self._make_optim_target(network,
                                                            optim_name,
                                                            **kwargs)
        return network, optimizer, target_network

    def _make_optim_target(self, network, optim_name, **kwargs):
        self.init_params(network)
        optimizer = self.get_optimizer_by_name(
            modules=[network], optim_name=optim_name, lr=kwargs['lr'])
        target_network = copy.deepcopy(network)
        for param in target_network.parameters():
            param.requires_grad = False

        return optimizer, target_network

    def act(self, obs: np.array) -> np.ndarray:
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = self.enc_ac(obs)
        action = self.mu(obs)
        if self.discrete:
            action = F.softmax(action, dim=-1)
        else:
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()
        return action

    def val(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        # TODO: Currently not using func; remove later
        obs = torch.form_numpy(obs).float().to(self.device)
        obs = self.enc_cr(obs)
        act = torch.from_numpy(obs).float()
        value = self.q(torch.cat([obs, act], dim=-1))
        value = value.detach().cpu().numpy()

        return value

    def forward(self, obs) -> Distribution:
        obs = obs.to(self.device)
        state_ac = self.enc_ac(obs)
        state_cr = self.enc_cr(obs)
        action = self.mu(state_ac)
        # act = ac_dist.mean
        value = self.q(state_cr, action)

        return action, value

    def step_ac(self, loss_ac: torch.tensor):
        if self.enc_ac.net is not None:
            self.optim_enc_ac.zero_grad()
        self.optim_ac.zero_grad()

        loss_ac.backward()
        torch.nn.utils.clip_grad_norm_(
            self.mu.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            self.enc_ac.parameters(), self.grad_clip)

        self.optim_ac.step()
        if self.enc_ac.net is not None:
            self.optim_enc_ac.step()

    def step_cr(self, loss_cr: torch.tensor):
        if self.enc_cr.net is not None:
            self.optim_enc_cr.zero_grad()
        self.optim_cr.zero_grad()

        loss_cr.backward()
        torch.nn.utils.clip_grad_norm_(
            self.mu.parameters(), self.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            self.enc_cr.parameters(), self.grad_clip)

        self.optim_cr.step()
        if self.enc_cr.net is not None:
            self.optim_enc_cr.step()

    def update_trg(self):
        self.polyak_update(self.mu, self.mu_trg, self.polyak)
        self.polyak_update(self.q, self.q_trg, self.polyak)
        if self.enc_ac.net is not None and self.enc_cr.net is not None:
            self.polyak_update(self.enc_ac, self.enc_ac_trg, self.polyak)
            self.polyak_update(self.enc_cr, self.enc_cr_trg, self.polyak)

    def save(self, save_dir):
        torch.save(self.mu.state_dict(), os.path.join(save_dir, 'actor.pt'))
        torch.save(self.q.state_dict(), os.path.join(save_dir, 'critic.pt'))
        print(f'model saved in {save_dir}')

    def load(self, load_dir):
        ckpt = torch.load(
            load_dir,
            map_location=self.device
        )
        self.model.mu.load_state_dict(ckpt)
        self.model.q.load_state_dict(ckpt)


class DDPGAgent(Agent):
    def __init__(self,
                 model: DDPGModel,
                 update_interval: int = 1,
                 train_interval: int = 1,
                 num_epochs: int = 1,
                 buffer_cls: ReplayBuffer = ExperienceReplay,
                 buffer_size: int = int(1e6),
                 buffer_kwargs: dict = None,
                 batch_size: int = 128,
                 explore: bool = True,
                 action_low: np.ndarray = None,
                 action_high: np.ndarray = None,
                 loss_func_ac: Callable = loss_func_ac,
                 loss_func_cr: Callable = loss_func_cr,
                 save_interval: int = int(1e5),
                 eps: float = 1e-5,
                 gamma: float = 0.99,
                 log_interval: int = int(1e3),
                 train_after: int = int(1e3),
                 update_after: int = int(1e3),
                 **kwargs):
        self.save_interval = save_interval
        if loss_func_cr is None:
            self.loss_func_cr = loss_func_cr
        if loss_func_ac is None:
            self.loss_func_ac = loss_func_ac

        self.buffer_size = buffer_size
        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.config.buffer_size}

        super().__init__(model,
                         update_interval,
                         num_epochs,
                         buffer_cls,
                         buffer_kwargs)

        self.model = model
        self.train_interval = train_interval
        self.eps = eps
        self.explore = explore
        self.action_low = action_low
        self.action_high = action_high
        self.loss_func_ac = loss_func_ac
        self.loss_func_cr = loss_func_cr
        self.gamma = gamma
        self.log_interval = log_interval
        self.logger = kwargs['logger']
        self.log_dir = self.logger.log_dir
        self.train_after = train_after
        self.update_after = update_after

    def act(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) in (1, 3):
            np.expand_dims(obs, axis=0)
        action = self.model.act(obs)
        if self.model.discrete:
            if self.explore:
                _action = []
                for ac in action:
                    _action.append(np.random.choice(np.arange(len(ac)), p=ac))
                action = np.array(_action)
            else:
                action = np.max(action, axis=-1)
        else:
            if self.explore:
                action += self.eps * np.random.randn(*action.shape)

        action = np.clip(action, self.action_low, self.action_high)

        return action

    def step(self, s, a, r, d, s_):
        self.collect(s, a, r, d, s_)
        info = {}
        if (self.curr_step % self.train_interval == 0 and
                self.curr_step > self.train_after):
            info = self.train()
        if (self.curr_step % self.update_interval == 0 and
                self.curr_step > self.update_after):
            self.model.update_trg()
        if self.curr_step % self.save_interval == 0 and self.model.is_save:
            save_dir = os.path.join(
                self.log_dir, f'ckpt/{int(self.curr_step/1000)}k')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.model.save(save_dir)

        return info

    def train(self):
        for _ in range(self.num_epochs):
            batch = self.buffer.sample(self.batch_size)
            cl = self.loss_func_cr(batch, self.model, gamma=self.gamma)
            self.model.step_cr(cl)

            al = self.loss_func_ac(batch, self.model)
            self.model.step_ac(al)

        info = {
            'Loss/Actor': al.item(),
            'Loss/Critic': cl.item()
        }

        return info

    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        self.buffer.push(s, a, r, d, s_)
