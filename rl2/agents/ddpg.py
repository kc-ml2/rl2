from typing import Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.distributions import Distribution
import copy

from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.agents.base import Agent
from rl2.models.torch.base import PolicyBasedModel, TorchModel, ValueBasedModel
from rl2.buffers.base import ReplayBuffer, ExperienceReplay
from rl2.networks.torch.networks import MLP

# FIXME: Move loss function to somewhere else
# from rl2.loss import DDPGloss

# TODO: Implement Noise
# class Noise():
#     def __init__(self, action_shape):
#         self.action_shape = action_shape

#     def __call__(self) -> np.array:
#         return np.random.randn(self.action_shape)
#         # TODO: implement loss func


def loss_func(data, model: TorchModel, **kwargs) -> List[torch.tensor]:
    data = list(data)
    s, a, r, d, s_ = tuple(map(lambda x: torch.from_numpy(x).float().to(model.device), data))
    if model.enc is not None:
        s_ac = model.enc_ac(s)
        s_cr = model.enc_cr(s)
        s_ac_ = model.enc_ac_trg(s_)
        s_cr_ = model.enc_cr_trg(s_)

    with torch.no_grad():
        a_trg = model.mu_trg(s_ac_)
        q_trg = model.q_trg(torch.cat([s_cr_, a_trg], dim=-1))
        bellman_trg = r + kwargs['gamma'] * q_trg * (1-d)

    q_cr = model.q(torch.cat([s_cr, a], dim=-1))
    l_cr = F.smooth_l1_loss(q_cr, bellman_trg)

    mu = model.mu(s)
    if model.discrete:
        mu = torch.gumbel_softmax(mu, tau=1, hard=False)
    q_ac = model.q(torch.cat([s_cr, model.mu(s_ac)], dim=-1))
    l_ac = -q_ac.mean()

    loss = [l_ac, l_cr]

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
                 optim_ac: str = None,
                 optim_cr: str = None,
                 discrete: bool = False,
                 flatten: bool = False,
                 reorder: bool = False,
                 **kwargs):

        super().__init__(observation_shape, action_shape, **kwargs)

        self.config = EasyDict(DEFAULT_DDPG_CONFIG)
        if 'config' in kwargs.keys():
            self.config = EasyDict(kwargs['config'])
        if optim_ac is None:
            optim_ac = "torch.optim.Adam"
        if optim_cr is None:
            optim_cr = "torch.optim.Adam"

        obs_dim = observation_shape[0]
        self.enc_ac = DummyEncoder(encoder, reorder, flatten).to(self.device)
        self.enc_cr = DummyEncoder(encoder, reorder, flatten).to(self.device)
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
            self.enc_ac = self.enc_ac.to(self.device)
            self.enc_cr = self.enc_cr.to(self.device)
            self.enc_ac_trg = self.enc_ac_trg.to(self.device)
            self.enc_cr_trg = self.enc_cr_trg.to(self.device)
        self.mu, self.optim_ac, self.mu_trg = self._make_mlp_optim_target(
            actor, obs_dim, action_shape[0], optim_ac
        )
        self.q, self.optim_cr, self.q_trg = self._make_mlp_optim_target(
            critic, obs_dim + action_shape[0], 1, optim_cr
        )
        self.mu, self.q = self.mu.to(self.device), self.q.to(self.device)
        self.mu_trg, self.q_trg = self.mu_trg.to(self.device), self.q_trg.to(self.device)

    def _make_mlp_optim_target(self, network,
                               num_input, num_output, optim_name):
        if network is None:
            network = MLP(in_shape=num_input, out_shape=num_output,
                          hidden=[128, 128])
        optimizer, target_network = self._make_optim_target(network,
                                                            optim_name)
        return network, optimizer, target_network

    def _make_optim_target(self, network, optim_name):
        self.init_params(network)
        optimizer = self.get_optimizer_by_name(
            modules=[network], optim_name=optim_name)
        target_network = copy.deepcopy(network)
        for param in target_network.parameters():
            param.requires_grad = False

        return optimizer, target_network

    def act(self, obs: np.array) -> np.array:
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = self.enc_ac(obs)
        action = self.mu(obs)
        if self.discrete:
            action = F.softmax(action, dim=-1)
        action = action.detach().cpu().numpy()
        return action

    def val(self, obs: np.array, act: np.array) -> np.array:
        # TODO: Currently not using func; remove later
        obs = torch.form_numpy(obs).float().to(self.device)
        obs = self.enc_cr(obs)
        act = torch.from_numpy(obs).float()
        value = self.q(torch.cat(obs, act))
        value = value.detach().cpu().numpy()

        return value

    def forward(self, obs) -> Distribution:
        obs = obs.to(self.device)
        state_ac = self.enc_ac(obs)
        state_cr = self.enc_cr(obs)
        ac_dist = self.mu(state_ac)
        # act = ac_dist.mean
        val_dist = self.q(state_cr, ac_dist)

        return ac_dist, val_dist

    def step_cr(self, loss_cr: torch.tensor):
        if self.enc_cr.net is not None:
            self.optim_enc_cr.zero_grad()
        self.optim_cr.zero_grad()
        loss_cr.backward()
        self.optim_cr.step()
        if self.enc_cr.net is not None:
            self.optim_enc_cr.step()

    def step_ac(self, loss_ac: torch.tensor):
        if self.enc_ac.net is not None:
            self.optim_enc_ac.zero_grad()
        self.optim_ac.zero_grad()
        loss_ac.backward()
        self.optim_ac.step()
        if self.enc_ac.net is not None:
            self.optim_enc_ac.step()

    def update_trg(self):
        self.polyak_update(self.mu, self.mu_trg, alpha=self.config.polyak)
        self.polyak_update(self.q, self.q_trg, alpha=self.config.polyak)
        if self.enc_ac.net is not None and self.enc_cr.net is not None:
            self.polyak_update(self.enc_ac, self.enc_ac_trg,
                               alpha=self.config.polyak)
            self.polyak_update(self.enc_cr, self.enc_cr_trg,
                               alpha=self.config.polyak)

    # def update_trg(self):
    #     polyak_update(self.mu, self.mu_trg, tau=self.tau)
    #     polyak_update(self.q, self.q_trg, tau=self.tau)

    def save(self):
        # torch.save(os.path.join(save_dir, 'encoder_ac.pt'))
        # torch.save(os.path.join(save_dir, 'encoder_cr.pt'))
        # torch.save(os.path.join(save_dir, 'actor.pt'))
        # torch.save(os.path.join(save_dir, 'critic.pt'))
        pass

    def load(self):
        pass


class DDPGAgent(Agent):
    def __init__(self,
                 model: DDPGModel,
                 update_interval: int = 1,
                 train_interval: int = 1,
                 num_epochs: int = 1,
                 buffer_cls: ReplayBuffer = ExperienceReplay,
                 buffer_kwargs: dict = None,
                 explore: bool = True,
                 **kwargs):
        # TODO: process config
        # config = kwargs['config']
        self.config = EasyDict(DEFAULT_DDPG_CONFIG)
        if 'config' in kwargs.keys():
            self.config = EasyDict(kwargs['config'])

        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.config.buffer_size}

        super().__init__(model,
                         update_interval,
                         num_epochs,
                         buffer_cls,
                         buffer_kwargs)

        self.model = model
        self.train_interval = train_interval
        # TODO: change to noise func or class for eps scheduling
        self.loss_func = loss_func
        self.eps = 0.01
        self.explore = explore

    def act(self, obs: np.array) -> np.array:
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

        return action

    def step(self, s, a, r, d, s_):
        self.collect(s, a, r, d, s_)
        if self.curr_step % self.train_interval == 0 and self.buffer.is_full:
            self.train()
        if self.curr_step % self.update_interval == 0 and self.buffer.is_full:
            self.model.update_trg()

    def train(self):
        batch = self.buffer.sample(self.config.batch_size)
        loss: List[Any] = self.loss_func(
            batch, self.model, gamma=self.config.gamma)
        self.model.step(loss)

        # FIXME: tmp logging; remove later
        if self.curr_step % self.config.log_interval == 0:
            log = list(map(lambda x: x.detach().cpu().numpy(), loss))
            print(f"ac_loss: {log[0]}, cr_loss: {log[1]}")

    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        self.buffer.push(s, a, r, d, s_)
