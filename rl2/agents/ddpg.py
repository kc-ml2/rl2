from typing import Any, List, OrderedDict, Callable
import torch
import numpy as np
from torch import float32, nn
from torch import optim
import torch.nn.functional as F
from importlib import import_module
from rl2.models.torch.base import PolicyBasedModel, TorchModel, ValueBasedModel
from torch.distributions import Distribution
import copy

from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.agents.base import Agent
from rl2.models.torch.base import PolicyBasedModel, ValueBasedModel
from rl2.buffers.base import ReplayBuffer
from rl2.networks.torch.networks import MLP

# from rl2.loss import DDPGloss
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
    s, a, r, d, s_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model), data))
    a_trg = model.mu_trg(s_)
    bellman_trg = r + kwargs.gamma * \
        model.q_trg(s, a_trg) * (1-d)
    q = model.q(s, model.mu(s))

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
                 observation_shape,
                 action_shape,
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
            critic = MLP(in_shape=(observation_shape[0] + action_shape[0]),
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

        for p_mu, p_q in zip(self.mu_trg.parameters(), self.q_trg.parameters()):
            p_mu.requires_grad = False
            p_q.requires_grad = False

    def act(self, obs: np.array) -> np.array:
        action = self.mu(torch.from_numpy(obs).float())
        action = action.detach().cpu().numpy()

        return action

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
        if buffer_kwargs is None:
            buffer_kwargs = {'size': 10}

        super().__init__(model,
                         update_interval,
                         num_epochs,
                         buffer_cls,
                         buffer_kwargs,)
        # self.config = kwargs['config']
        self.model = model
        self.loss_func = loss_func
        self.eps = 0.1
        self.explore = True
        # TODO: change to noise func or class

    def act(self, obs: np.array) -> np.array:
        act = self.model.act(obs)
        if self.explore:
            act += self.eps * np.random.randn(*act.shape)

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
