import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Distribution
from rl2.agents.base import Agent
from rl2.buffers.base import ExperienceReplay, ReplayBuffer
from rl2.models.torch.base import TorchModel
from rl2.models.torch.base import InjectiveBranchModel, BranchModel


def loss_func_ac(data, model, **kwargs):
    s = torch.from_numpy(data[0]).float().to(model.device)
    mu = model.mu(s).mean
    if not model.discrete:
        mu = torch.tanh(mu)
    q_val = model.q(s, mu).mean
    loss = -q_val.mean()

    return loss


def loss_func_cr(data, model, **kwargs):
    s, a, r, d, s_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
    )
    with torch.no_grad():
        a_trg = model.mu.forward_trg(s_).mean
        if not model.discrete:
            a_trg = torch.tanh(a_trg)
        v_trg = model.q.forward_trg(s_, a_trg).mean
        bellman_trg = r + kwargs['gamma'] * v_trg * (1-d)

    q = model.q(s, a).mean
    loss = F.smooth_l1_loss(q, bellman_trg)

    return loss


def loss_func_cr_mix(data, model, **kwargs):
    sample_size = 16
    eps = 1e-1
    num_action = data[1].shape[-1]
    samples = np.random.dirichlet(np.ones(num_action), sample_size)
    a = np.eye(num_action)
    s = data[0]
    num_batch = data[0].shape[0]

    s, a, samples = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device),
            [s, a, samples])
    )
    with torch.no_grad():
        s_r = s.repeat_interleave(num_action, dim=0)
        a = a.repeat(num_batch, 1)

        q = model.q(s_r, a).mean
        q = q.reshape(num_batch, -1)

        q_min = q.min(-1, keepdim=True)[0]
        q_mixed = (q * samples).sum(-1, keepdim=True)
        noisy_samples = samples + eps * torch.randn_like(samples)
    q_int = model.q(s, samples).mean
    q_ext = model.q(s, noisy_samples).mean
    loss = F.mse_loss(q_int, q_mixed) + F.mse_loss(q_ext, q_min)

    return loss


class DDPGModel(TorchModel):
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
                 encoded_dim: int = 64,
                 optim_ac: str = 'torch.optim.Adam',
                 optim_cr: str = 'torch.optim.Adam',
                 lr_ac: float = 1e-4,
                 lr_cr: float = 1e-4,
                 grad_clip: float = 1e-2,
                 polyak: float = 0.995,
                 discrete: bool = False,
                 flatten: bool = False,
                 reorder: bool = False,
                 recurrent: bool = False,
                 **kwargs):

        super().__init__(observation_shape, action_shape, **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape
        self.encoded_dim = encoded_dim
        self.recurrent = recurrent

        if optim_ac is None:
            optim_ac = self.optim_ac
        if optim_cr is None:
            optim_cr = self.optim_cr

        self.grad_clip = grad_clip
        self.discrete = discrete

        self.lr_ac = lr_ac
        self.lr_cr = lr_cr
        self.polyak = polyak
        self.is_save = kwargs.get('is_save', False)

        self.mu = BranchModel(observation_shape, action_shape,
                              encoded_dim=encoded_dim,
                              optimizer=optim_ac,
                              lr=lr_ac,
                              grad_clip=grad_clip,
                              make_target=True,
                              discrete=discrete,
                              deterministic=True,
                              recurrent=recurrent,
                              reorder=reorder,
                              flatten=flatten,
                              default=False,
                              head_depth=2,
                              **kwargs)

        self.q = InjectiveBranchModel(observation_shape, (1,), action_shape,
                                      encoded_dim=encoded_dim,
                                      optimizer=optim_cr,
                                      lr=lr_cr,
                                      grad_clip=grad_clip,
                                      make_target=True,
                                      discrete=False,
                                      deterministic=True,
                                      recurrent=recurrent,
                                      reorder=reorder,
                                      flatten=flatten,
                                      default=False,
                                      head_depth=2,
                                      **kwargs)
        self.init_params(self.mu)
        self.init_params(self.q)

    @TorchModel.sharedbranch
    def forward(self, obs, **kwargs) -> Distribution:
        obs = obs.to(self.device)
        state_ac = self.enc_ac(obs)
        state_cr = self.enc_cr(obs)
        action_dist = self.mu(state_ac, **kwargs)
        if self.recurrent:
            action_dist = action_dist[0]
        value_dist = self.q(state_cr, action_dist.mean, **kwargs)
        if self.recurrent:
            value_dist = value_dist[0]

        return action_dist, value_dist

    def act(self, obs: np.array) -> np.ndarray:
        action_dist, hidden = self._infer_from_numpy(self.mu, obs)
        action = action_dist.mean
        if not self.discrete:
            action = torch.tanh(action)
        action = action.detach().cpu().numpy()

        info = {}
        if self.recurrent:
            info['hidden'] = hidden

        return action, info

    def val(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        # TODO: Currently not using func; remove later
        value_dist, hidden = self._infer_from_numpy(self.q, obs, act)
        value = value_dist.mean
        value = value.detach().cpu().numpy()

        info = {}
        if self.recurrent:
            info['hidden'] = hidden

        return value

    def update_trg(self):
        self.mu.update_trg(alpha=self.polyak)
        self.q.update_trg(alpha=self.polyak)

    def save(self, save_dir):
        # torch.save(self.enc_ac.state_dict(),
        #            os.path.join(save_dir, 'enc_ac.pt'))
        # torch.save(self.enc_cr.state_dict(),
        #            os.path.join(save_dir, 'enc_ac.pt'))
        torch.save(self.mu.state_dict(), os.path.join(save_dir, 'actor.pt'))
        torch.save(self.q.state_dict(), os.path.join(save_dir, 'critic.pt'))
        print(f'model saved in {save_dir}')

    def load(self, load_dir_ac, load_dir_cr):
        # TODO: load pretrained model
        # ckpt = torch.load(
        #     os.path.join(load_dir, 'enc_ac.pt'),
        #     map_location=self.device
        # )
        # self.enc_ac.load_state_dict(ckpt)
        # self.mu.load_state_dict(ckpt)
        # self.enc_cr.load_state_dict(ckpt)
        # self.q.load_state_dict(ckpt)
        ckpt_ac = torch.load(load_dir_ac, map_location=self.device)
        self.mu.load_state_dict(ckpt_ac)
        ckpt_cr = torch.load(load_dir_cr, map_location=self.device)
        self.q.load_state_dict(ckpt_cr)


class DDPGAgent(Agent):
    def __init__(self,
                 model: DDPGModel,
                 update_interval: int = 1000,
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
                 save_interval: int = int(1e6),
                 eps: float = 0.1,
                 gamma: float = 0.99,
                 log_interval: int = int(1e3),
                 train_after: int = int(1e3),
                 update_after: int = int(1e3),
                 **kwargs):
        if loss_func_cr is None:
            self.loss_func_cr = loss_func_cr
        if loss_func_ac is None:
            self.loss_func_ac = loss_func_ac
        self.loss_func_cr_mix = loss_func_cr_mix

        self.buffer_size = buffer_size
        if buffer_kwargs is None:
            buffer_kwargs = {'size': self.buffer_size,
                             'state_shape': model.observation_shape,
                             'action_shape': model.action_shape}

        super().__init__(model=model,
                         train_interval=train_interval,
                         num_epochs=num_epochs,
                         buffer_cls=buffer_cls,
                         buffer_kwargs=buffer_kwargs)

        # Set intervals
        self.train_interval = train_interval
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.log_interval = log_interval

        # set hyperparams
        self.train_after = train_after
        self.update_after = update_after
        self.batch_size = batch_size
        self.eps = eps
        self.explore = explore
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.logger = kwargs.get('logger')
        if self.logger:
            self.log_dir = self.logger.log_dir

        # Set loss functions
        self.loss_func_ac = loss_func_ac
        self.loss_func_cr = loss_func_cr

        # For recurrent intermediate representation
        self.done = False
        self.model._init_hidden(self.done)
        self.hidden = self.model.hidden
        self.pre_hidden = self.hidden

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, info = self.model.act(obs)
        if self.model.discrete:
            self.action_param = action
            # action = np.argmax(action, axis=-1)
            action = np.random.choice(action.shape[-1], p=action.squeeze())
            # if self.explore:
            #     if np.random.random() < self.eps:
            #         action = np.random.randint(self.model.action_shape[0],
            #                                    size=action.shape)
            # action = action.squeeze()
        else:
            if self.explore:
                action += self.eps * np.random.randn(*action.shape)
            action = np.clip(action, self.action_low, self.action_high)

        if self.model.recurrent:
            self.hidden = info['hidden']

        return action

    def step(self, s, a, r, d, s_):
        self.curr_step += 1
        if self.model.discrete:
            a = self.action_param
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
            # cl_mix = self.loss_func_cr_mix(batch, self.model)
            self.model.q.step(cl)
            # self.model.q.step(cl+cl_mix)

            al = self.loss_func_ac(batch, self.model)
            self.model.mu.step(al)

        info = {
            'Loss/Actor': al.item(),
            'Loss/Critic': cl.item()
        }

        return info

    def collect(self, s, a, r, d, s_):
        # if self.model.discrete:
        #     a = np.eye(self.model.action_shape[0])[a]
        self.buffer.push(s, a, r, d, s_)
