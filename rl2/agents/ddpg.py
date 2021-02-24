# from marlenv.marlenv.envs.utils import observation_space
# from marlenv.marlenv.envs.utils import action_space
from typing import Any, List, OrderedDict, Callable
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from importlib import import_module
from rl2.models.torch.base import PolicyBasedModel, TorchModel, ValueBasedModel
from torch.distributions import Distribution
import copy

from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.agents.base import Agent
from rl2.models.torch.base import PolicyBasedModel, ValueBasedModel
# from rl2.models.torch.dpg import DPGModel
# from rl2.models.torch.actor_critic import ActorCriticModel
from rl2.buffers.base import ReplayBuffer
from rl2.networks.torch.networks import MLP
from rl2.networks.torch.distributional import ScalarHead
# from rl2.loss import DDPGloss

# from rl2.utils import Noise

# TODO: Implement Noise


class Noise():
    def __init__(self, action_shape):
        self.action_shape = action_shape

    def __call__(self) -> np.array:
        return np.random.randn(self.action_shape)
        # TODO: implement loss func


def polyak_update(source, trg, tau=0.995):
    # TODO: Implement polyak step update
    for p, pt in zip(source.parameters(), trg.parameters()):
        pt.data.copy_(tau * pt.data + (1-tau)*p.data)


def loss_func(data,
              model: TorchModel,
              **kwargs) -> List[torch.tensor]:
    a_trg = model.mu_trg(data.s_)
    bellman_trg = data.r + kwargs.gamma * \
        model.q_trg(data.s, a_trg) * (1-data.d)
    q = model.q(data.s, model.mu(data.s))

    l_ac = F.smooth_l1_loss(q, bellman_trg)
    l_cr = -q.mean()

    loss = [l_ac, l_cr]

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
            # FIXME: Acition_dim management
            self.encoder_cr = MLP(in_shape=(input_shape + action_dim[0]),
                                  out_shape=enc_dim)

        self.actor = ScalarHead(
            input_size=enc_dim,
            out_size=1)
        self.critic = ScalarHead(
            input_size=enc_dim,
            out_size=1)

        self.mu = nn.Sequential(
            self.encoder_ac,
            self.actor
        )

        self.q = nn.Sequential(
            self.encoder_cr,
            self.critic
        )

        if optim_ac is None:
            self.optim_ac = "torch.optim.Adam"
        if optim_cr is None:
            self.optim_cr = "torch.optim.Adam"

        self.optim_ac = self.get_optimizer_by_name(
            modules=self.mu, optim_name=self.optim_ac)
        self.optim_cr = self.get_optimizer_by_name(
            modules=self.q, optim_name=self.optim_cr)  # , optim_kwargs=kwargs['optim_kwargs_cr'])

        self.mu_trg = copy.deepcopy(self.mu)
        self.q_trg = copy.deepcopy(self.q)

        for p_mu, p_q in zip(self.mu_trg.parameters(), self.q_trg.parameters()):
            p_mu.requires_grad = False
            p_q.requires_grad = False

    def act(self, obs: np.array) -> np.array:
        ac_dist = self.mu(torch.as_tensor(obs))
        act = ac_dist.mean
        act.numpy()

        return act

    def val(self, obs, act) -> Distribution:
        val_dist = self.q(obs, act)
        val = val_dist.mean
        val.numpy()

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
                 update_interval=1,
                 observation_shape=None,
                 action_shape=None,
                 num_epochs=1,
                 # FIXME: handle device in model class
                 device=None,
                 buffer_cls=ReplayBuffer,
                 buffer_kwargs=None,
                 **kwargs):
        #  loss_func: Callable[["data", "model"], List[torch.tensor]],
        #  noise: Any,
        # config = kwargs['config']

        # TODO: process config
        assert observation_shape is not None, "obs error"
        assert action_shape is not None, "ac error"
        if device is None:
            device = 'cpu'
        if buffer_kwargs is None:
            buffer_kwargs = {'size': 10}

        super().__init__(model,
                         update_interval,
                         observation_shape,
                         action_shape,
                         num_epochs,
                         device,
                         buffer_cls,
                         buffer_kwargs,)
        self.config = kwargs['config']
        self.model = model
        self.loss_func = loss_func
        self.noise = Noise(action_shape)

    def act(self, obs: np.array) -> np.array:
        act = self.model.act(obs)
        if self.explore:
            act += self.noise  # * self.config.eps

        return act

    def step(self, s, a, r, d, s_):
        self.collect(s, a, r, d, s_)
        if self.curr_step % self.train_interval == 0:
            self.train()
        if self.curr_step % self.trg_update_interval == 0:
            self.model.update_trg()

    def train(self):
        batch = self.buffer.sample()
        loss: List[Any] = self.loss_func(batch, self.model)
        self.model.step(loss)

    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        self.buffer.push(s, a, r, d, s_)


# if __name__ == "__main__":
#     m=DDPGAgent(input_shape=5, enc_dim=128)
#     m.mu
