import copy
import os
from pathlib import Path
from typing import Callable, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl2.agents.base import Agent
from rl2.buffers.base import ExperienceReplay, ReplayBuffer
from rl2.models.torch.base import ValueBasedModel
from rl2.networks.torch.networks import MLP


def loss_func(data, model, **kwargs):
    s, a, r, d, s_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
    )
    if model.enc is not None:
        s = model.enc(s)
        s_ = model.enc_trg(s_)

    _batch_size = s.shape[0]

    with torch.no_grad():
        a_dim = model.action_shape[0]
        actions = torch.arange(0, a_dim).repeat(_batch_size).view(
            _batch_size, a_dim, -1).float().to(model.device)
        # actions = torch.arange(0, a_dim).unsqueeze(-1).float().to(model.device)
        s__ = s_.repeat(1, a_dim).view(_batch_size, a_dim, -1)
        a_trg = torch.argmax(model.q_trg(
            torch.cat([s__, actions], dim=-1)), dim=1).float()  # .unsqueeze(-1)

        v_trg = model.q_trg(torch.cat([s_, a_trg], dim=-1))
        bellman_trg = r + kwargs['gamma'] * v_trg * (1-d)

    q = model.q(torch.cat([s, a], dim=-1))
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


class DQNModel(ValueBasedModel):
    """
    predefined model
    (same one as original paper)
    """

    def __init__(self,
                 observation_shape,
                 action_shape,
                 q_network: torch.nn.Module = None,
                 encoder: torch.nn.Module = None,
                 encoder_dim: int = None,
                 optim: str = 'torch.optim.Adam',
                 lr: float = 1e-4,
                 grad_clip: float = 1e-2,
                 polyak: float = float(0),
                 discrete: bool = True,
                 flatten: bool = False,
                 reorder: bool = False,
                 **kwargs):
        # FIXME: handling action shape for discrete action -> error in buffer
        super().__init__(observation_shape, action_shape, **kwargs)
        self.action_shape = (1,)

        if optim is None:
            optim = self.optim

        self.grad_clip = grad_clip

        self.lr = lr
        self.polyak = polyak
        self.is_save = kwargs.get('is_save', False)

        obs_dim = observation_shape[0]
        self.enc = DummyEncoder(encoder, reorder, flatten).to(self.device)
        self.enc_trg = copy.deepcopy(self.enc)
        self.discrete = discrete
        if len(observation_shape) == 3:
            assert encoder is not None, 'Must provide an encoder for 2d input'
            assert encoder_dim is not None

            self.optim_enc, self.enc_trg = self._make_optim_target(
                self.enc, optim
            )
            obs_dim = encoder_dim
            self.enc, self.enc_trg = map(lambda x: x.to(self.device),
                                         [self.enc,  self.enc_trg])

        self.q, self.optim, self.q_trg = self._make_mlp_optim_target(
            q_network, obs_dim + 1, 1, optim, lr=self.lr)

        self.q, self.q_trg = map(lambda x: x.to(self.device),
                                 [self.q, self.q_trg])

    def _make_mlp_optim_target(self, network,
                               num_input, num_output, optim_name, **kwargs):
        lr = kwargs.get('lr', 1e-4)  # Form a optim args elsewhere
        if network is None:
            network = MLP(in_shape=num_input, out_shape=num_output,
                          hidden=[128, 128])
        optimizer, target_network = self._make_optim_target(network,
                                                            optim_name,
                                                            lr=lr)
        return network, optimizer, target_network

    def _make_optim_target(self, network, optim_name, **optim_args):
        self.init_params(network)
        optimizer = self.get_optimizer_by_name(
            modules=[network], optim_name=optim_name, **optim_args)
        target_network = copy.deepcopy(network)
        for param in target_network.parameters():
            param.requires_grad = False

        return optimizer, target_network

    def act(self, obs: np.array) -> np.ndarray:
        obs = torch.from_numpy(obs).float().to(self.device)
        a_dim = self.action_shape[0]
        actions = torch.arange(0, a_dim).view(
            1, a_dim, -1).float().to(self.device)
        state = self.enc(obs)
        state = state.repeat(1, a_dim).view(1, a_dim, -1)

        if np.random.random() > self.eps:
            action = torch.argmax(
                self.q(torch.cat([state, actions], dim=-1))).unsqueeze(-1)
        else:
            action = torch.randint(self.action_shape[0], (1,))

        action = action.detach().cpu().numpy()

        return action

    def val(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        # TODO: Currently not using func; remove later
        obs = torch.from_numpy(obs).float().to(self.device)
        state = self.enc(obs)
        act = torch.from_numpy(act).float().to(self.device)
        value = self.q(torch.cat([state, act], dim=-1))
        value = value.detach().cpu().numpy()

        return value

    def forward(self, obs: np.ndarray) -> np.ndarray:
        state = self.enc(obs)
        action = self.act(state)
        value = self.val(state, action)

        return action, value

    def step(self, loss: torch.tensor):
        if self.enc.net is not None:
            self.optim_enc.zero_grad()
        self.optim.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q.parameters(), self.grad_clip)

        self.optim.step()

        if self.enc.net is not None:
            torch.nn.utils.clip_grad_norm_(
                self.enc.parameters(), self.grad_clip)

            self.optim_enc.step()

    def update_trg(self):
        self.polyak_update(self.q, self.q_trg, self.polyak)
        if self.enc.net is not None:
            self.polyak_update(self.enc, self.enc_trg, self.polyak)

    def save(self, save_dir):
        torch.save(self.q.state_dict(), os.path.join(save_dir, 'q_network.pt'))
        print(f'model saved in {save_dir}')

    def load(self, load_dir):
        ckpt = torch.load(
            load_dir,
            map_location=self.device
        )
        self.model.q.load_state_dict(ckpt)


class DQNAgent(Agent):
    def __init__(self,
                 model: DQNModel,
                 update_interval: int = 1,
                 train_interval: int = 1,
                 num_epochs: int = 1,
                 buffer_cls: Type[ReplayBuffer] = ExperienceReplay,
                 buffer_size: int = int(1e6),
                 buffer_kwargs: dict = None,
                 batch_size: int = 128,
                 explore: bool = True,
                 loss_func: Callable = loss_func,
                 save_interval: int = int(1e5),
                 eps: float = 1e-5,
                 gamma: float = 0.99,
                 log_interval: int = int(1e3),
                 train_after: int = int(1e3),
                 update_after: int = int(1e3),
                 **kwargs):
        self.save_interval = save_interval
        if loss_func is None:
            self.loss_func = loss_func

        self.buffer_size = buffer_size
        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.buffer_size}

        super().__init__(model,
                         update_interval,
                         num_epochs,
                         buffer_cls,
                         buffer_kwargs)

        self.train_interval = train_interval
        self.batch_size = batch_size
        # FIXME: handling exploration
        self.model.eps = eps
        self.explore = explore
        self.loss_func = loss_func
        self.gamma = gamma
        self.log_interval = log_interval
        self.logger = kwargs['logger']
        self.log_dir = self.logger.log_dir
        self.train_after = train_after
        self.update_after = update_after

    def act(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) in (1, 3):
            obs = np.expand_dims(obs, axis=0)
        action = self.model.act(obs)
        # FIXME: expand_dims?
        return action[0]

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
            loss = self.loss_func(batch, self.model, gamma=self.gamma)
            self.model.step(loss)

        info = {
            'Loss/Q_network': loss.item(),
        }

        return info

    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        self.buffer.push(s, a, r, d, s_)
