from typing import Any, List, OrderedDict
import torch
import numpy as np
from torch import nn
from importlib import import_module
from rl2.models.torch.base import PolicyBasedModel, ValueBasedModel
from torch.distributions import Distribution

from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.agents.base import Agent
from rl2.models.torch.base import PolicyBasedModel, ValueBasedModel
# from rl2.models.torch.dpg import DPGModel
# from rl2.models.torch.actor_critic import ActorCriticModel
from rl2.buffers.base import ReplayBuffer
from rl2.networks.torch.networks import MLP
from rl2.networks.torch.distributional import ScalarHead
# from rl2.loss import DDPGloss

# from rl2.utils import noise

# TODO: implement loss func


def loss_func_ac(self, data: 'batch'):
    losses = []
    s, a, r, d, s_ = data['obs']
    a_trg = self.agent.mu_trg(s)
    backup = r + gamma * self.agent.q_trg(s, a_trg)
    loss = (backup - self.q(s, a)) ** 2
    loss = loss.mean()

    val_dist = self.agent.q_trg(o, a)
    q = val_dist.mean()


def loss_func_cr(self, data: 'batch'):
    obs = data['obs']
    act = data['act']
    trg = rew + gamma*self.mu_trg(obs, act)
    q = self.model.infer(o, a)
    self.critic()

    pass


def loss_func(self, data, **kwargs):
    obs = data['obs']
    ac_dist, val_dist = self.model.infer(obs)
    vals = val_dist.mean
    ac = ac_dist.mean

    ac_loss = self.loss_func_ac(**kwargs)
    val_loss = self.loss_func_cr(**kwargs)

    loss = [ac_loss, val_loss]
    raise NotImplementedError

    return loss


class DDPGModel(PolicyBasedModel, ValueBasedModel):
    """
    predefined model
    (same one as original paper)
    """

    def __init__(self,
                 input_shape,
                 enc_dim,
                 action_dim,
                 enc_ac: torch.nn.Module = None,
                 enc_cr: torch.nn.Module = None,
                 optim_ac: str = None,
                 optim_cr: str = None,
                 **kwargs):

        super().__init__(input_shape, **kwargs)
        # config = kwargs['config']
        if enc_ac is None:
            self.encoder_ac = MLP(in_shape=input_shape,
                                  out_shape=enc_dim)
        if enc_cr is None:
            self.encoder_cr = MLP(in_shape=(input_shape + action_dim),
                                  out_shape=enc_dim)

        self.actor = ScalarHead(
            input_size=self.encoder_ac.out_shape,
            out_size=1)
        self.critic = ScalarHead(
            input_size=self.encoder_cr.out_shape,
            out_size=1)

        self.mu = nn.Sequential(OrderedDict([
            ('enc_ac', self.encoder_ac),
            ('ac', self.actor)
        ]))

        self.q = nn.Sequential(OrderedDict([
            ('enc_cr', self.encoder_cr),
            ('cr', self.critic)
        ]))

        self.optim_ac = self.get_optimizer_by_name(
            modules=self.mu, optim_name=optim_ac, **kwargs.optim_kwargs_ac)
        self.optim_cr = self.get_optimizer_by_name(
            modules=self.q, optim_name=optim_cr, **kwargs.optim_kwargs_cr)

    def act(self, obs: np.array) -> np.array:
        ac_dist = self.mu(obs)
        act = ac_dist.mean
        act.numpy()

        return act

    def val(self, obs, act) -> Distribution:
        val_dist = self.q(obs, act)
        val = val_dist.mean

        return val

    def forward(self, obs) -> Distribution:
        ac_dist = self.mu(obs)
        act = ac_dist.mean
        val_dist = self.q(obs, act)

        return ac_dist, val_dist

    def step(self, loss: List[torch.tensor]):
        loss_ac, loss_cr = loss
        self.optim_ac.zero_grad()
        loss_ac.backward(retain_graph=True)
        self.optim_ac.step()

        self.optim_cr.zero_grad()
        loss_cr.backward()
        self.optim_cr.step()

        # TODO: Implement polyak step update
        raise NotImplementedError

    def save(self):
        # torch.save(os.path.join(save_dir, 'encoder_ac.pt'))
        # torch.save(os.path.join(save_dir, 'encoder_cr.pt'))
        # torch.save(os.path.join(save_dir, 'actor.pt'))
        # torch.save(os.path.join(save_dir, 'critic.pt'))
        pass

    def load(self):
        pass


class DDPGAgent(Agent):
    def __init__(self, model: DDPGModel, **kwargs):
        # config = kwargs['config']
        super().__init__(model, **kwargs)
        self.config = kwargs['config']
        self.buffer = ReplayBuffer()
        self.model = model

    def act(self, obs: np.array) -> np.array:
        act = self.model.act(obs)
        if self.explore:
            act: np.array += self.noise

        return act

    def step(self):
        if self.curr_step % self.update_interval == 0:
            loss: List[Any] = self.compute_loss(batch)
            self.model.step(loss)
            raise NotImplementedError

    def train(self):
        trg_mu = copy.deepcopy(self.model.mu)
        trg_q = copy.deepcopy(self.model.q)
        for p_mu, p_q in zip(trg_mu.parameters(), trg_q.parameters():
            p_mu.requires_grad=False
            p_q.requires_grad=False

        noise=Noise()

        for i_epoch in range(self.num_epochs):
            data=self.buffer.sample()
            loss=self.loss_func(data)
            self.model.step(loss)


    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        self.buffer.push(s, a, r, d, s_)


if __name__ == "__main__":
    m=DDPGAgent(input_shape=5, enc_dim=128)
    m.mu
