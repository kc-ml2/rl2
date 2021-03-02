from typing import Any, Callable, List
import torch
import numpy as np
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


def loss_func(data, model: TorchModel, **kwargs) -> List[torch.Tensor]:
    data = list(data)
    s, a, r, d, s_ = tuple(map(lambda x: torch.from_numpy(x).float(), data))

    a_trg = model.mu_trg(s_)
    v_trg = model.q_trg(torch.cat([s_, a_trg], dim=-1))
    bellman_trg = r + kwargs['gamma'] * v_trg * (1-d)

    q_cr = model.q(torch.cat([s, a], dim=-1))
    l_cr = F.mse_loss(q_cr, bellman_trg)
    # l_cr = F.smooth_l1_loss(q_cr, bellman_trg)

    q_ac = model.q(torch.cat([s, model.mu(s)], dim=-1))
    l_ac = -q_ac.mean()

    loss = [l_ac, l_cr]

    return loss


def loss_func_ac(data, model, **kwargs):
    data = list(data)
    s, a, r, d, s_ = tuple(map(lambda x: torch.from_numpy(x).float(), data))
    q_val = model.q(torch.cat([s, model.mu(s)], dim=-1))
    loss = -q_val.mean()
    return loss


def loss_func_cr(data, model, **kwargs):
    data = list(data)
    s, a, r, d, s_ = tuple(map(lambda x: torch.from_numpy(x).float(), data))
    q = model.q(torch.cat([s, a], dim=-1))
    a_trg = model.mu_trg(s_)
    v_trg = model.q_trg(torch.cat([s_, a_trg], dim=-1))
    bellman_trg = r + kwargs['gamma'] * v_trg * (1-d)
    loss = F.mse_loss(q, bellman_trg)

    return loss


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
            optim_ac,
            lr=self.config.lr_ac
        )
        self.q, self.optim_cr, self.q_trg = self._make_mlp_optim_target(
            critic,
            observation_shape[0] + action_shape[0],
            1,
            optim_cr,
            lr=self.config.lr_cr
        )

    def _make_mlp_optim_target(self, network,
                               num_input, num_output, optim_name, **kwargs):
        if network is None:
            network = MLP(in_shape=num_input, out_shape=num_output)
        optimizer = self.get_optimizer_by_name(
            modules=[network], optim_name=optim_name, lr=kwargs['lr'])
        target_network = copy.deepcopy(network)
        for param in target_network.parameters():
            param.requires_grad = False

        return network, optimizer, target_network

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = self.mu(torch.from_numpy(obs).float())
        action = torch.tanh(action)
        action = action.detach().cpu().numpy()

        return action

    def val(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        # TODO: Currently not using func; remove later
        obs = torch.from_numpy(obs).float()
        act = torch.from_numpy(obs).float()
        value = self.q(torch.cat([obs, act], dim=-1))
        value.detach().cpu().numpy()

        return value

    # def forward(self, obs) -> Distribution:
    #     ac_dist = self.mu(obs)
    #     act = ac_dist.mean
    #     val_dist = self.q(obs, act)

    #     return ac_dist, val_dist

    def step(self, data, loss_func: List[Callable]):
        loss_func_ac, loss_func_cr = loss_func
        # self.optim_ac.zero_grad()
        # loss_ac.backward(retain_graph=True)
        # # TODO: grad clip decorator clipS
        # torch.nn.utils.clip_grad_norm_(
        #     self.mu.parameters(), self.config.grad_clip)
        # self.optim_ac.step()

        # self.optim_cr.zero_grad()
        # loss_cr.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     self.q.parameters(), self.config.grad_clip)
        # self.optim_cr.step()
        self.optim_cr.zero_grad()
        loss_cr = loss_func_cr(data, self, gamma=self.config.gamma)
        loss_cr.backward()
        self.optim_cr.step()

        # Freeze Q-network
        for p in self.q.parameters():
            p.requires_grad = False

        self.optim_ac.zero_grad()
        loss_ac = loss_func_ac(data, self)
        loss_ac.backward()
        self.optim_ac.step()

        # Unfreeze Q-network
        for p in self.q.parameters():
            p.requires_grad = True

        return loss_ac, loss_cr

    def update_trg(self):
        self.polyak_update(self.mu, self.mu_trg, self.config.polyak)
        self.polyak_update(self.q, self.q_trg, self.config.polyak)

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
                 action_low: np.ndarray = None,
                 action_high: np.ndarray = None,
                 **kwargs):
        self.model = model
        self.config = self.model.config

        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.config.buffer_size,
                             'state_shape': self.model.observation_shape,
                             'action_shape': self.model.action_shape}

        super().__init__(model,
                         update_interval,
                         num_epochs,
                         buffer_cls,
                         buffer_kwargs)
        self.train_interval = train_interval
        self.loss_func = loss_func
        self.eps = self.config.eps
        self.explore = explore
        self.action_low = action_low
        self.action_high = action_high

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = self.model.act(obs)
        if self.explore:
            action += self.eps * np.random.randn(*action.shape)

        action = np.clip(action, self.action_low, self.action_high)

        return action

    def step(self, s, a, r, d, s_):
        self.collect(s, a, r, d, s_)
        if self.curr_step % self.train_interval == 0 and self.curr_step > self.config.batch_size:
            self.train()
        if self.curr_step % self.update_interval == 0 and self.curr_step > self.config.batch_size:
            self.model.update_trg()

    def train(self):
        for _ in range(self.config.num_epochs):
            batch = self.buffer.sample(self.config.batch_size)
            loss_func = [loss_func_ac, loss_func_cr]
            info = self.model.step(batch, loss_func)

        # FIXME: tmp logging; remove later
        if self.curr_step % self.config.log_interval == 0:
            log = list(map(lambda x: x.detach().cpu().numpy(), info))
            self.writer.add_scalar('Loss/Actor', log[0], self.curr_step)
            self.writer.add_scalar('Loss/Critic', log[1], self.curr_step)
            self.writer.flush()
            print(
                f"num_step: {self.curr_step}, ac_loss: {log[0]}, cr_loss: {log[1]}")

    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        self.buffer.push(s, a, r, d, s_)
