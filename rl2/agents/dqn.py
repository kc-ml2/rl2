import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Callable, Type
from rl2.agents.base import Agent
from rl2.buffers.base import ExperienceReplay, ReplayBuffer
from rl2.models.torch.base import BranchModel, ValueBasedModel
from rl2.agents.utils import LinearDecay


def loss_func(data, model, **kwargs):
    s, a, r, d, s_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
    )
    with torch.no_grad():
        if not model.double:
            v_trg = torch.max(model.q.forward_trg(s_).mean,
                              dim=-1, keepdim=True).values
        else:
            a_ = torch.argmax(model.q(s_).mean, dim=-1, keepdim=True)
            v_trg = torch.sum(
                (model.q.forward_trg(s_).mean * a_), dim=-1, keepdim=True)
        bellman_trg = r + kwargs['gamma'] * v_trg * (1-d)

    q = torch.sum((model.q(s).mean * a), dim=-1, keepdim=True)
    loss = F.smooth_l1_loss(q, bellman_trg)

    return loss


class DQNModel(ValueBasedModel):
    """
    predefined model
    (same one as original paper)
    """

    def __init__(self,
                 observation_shape,
                 action_shape,
                 double: bool = False,  # Double DQN if True
                 q_network: torch.nn.Module = None,
                 encoder: torch.nn.Module = None,
                 encoded_dim: int = 64,
                 optim: str = 'torch.optim.RMSprop',
                 lr: float = 1e-4,
                 grad_clip: float = 1e-2,
                 polyak: float = float(0),
                 discrete: bool = True,
                 flatten: bool = False,
                 reorder: bool = False,
                 **kwargs):
        super().__init__(observation_shape, action_shape, **kwargs)

        self.is_save = kwargs.get('is_save', False)
        self.discrete = discrete
        self.double = double
        self.rews_ep = 0

        # Set hyperparams
        self.polyak = polyak

        # Set default RMSprop optim_args
        if optim == 'torch.optim.RMSprop':
            kwargs.setdefault('optim_args', {'alpha': 0.95,
                                             'eps': 0.00001,
                                             'momentum': 0.0,
                                             'centered': True})

        self.q = BranchModel(observation_shape,
                             action_shape,
                             encoded_dim=encoded_dim,
                             optimizer=optim,
                             lr=lr,
                             grad_clip=grad_clip,
                             make_target=True,
                             discrete=discrete,
                             deterministic=True,
                             reorder=reorder,
                             flatten=flatten,
                             default=True,
                             head_depth=2,
                             **kwargs)
        self.init_params(self.q)

    def _clean(self):
        for mod in self.children():
            if isinstance(mod, BranchModel):
                mod.reset_encoder_memory()

    def act(self, obs: np.array) -> np.ndarray:
        obs = torch.from_numpy(obs).float().to(self.device)
        values = self.q(obs).mean
        action = torch.argmax(values)
        action = action.detach().cpu().numpy()
        self._clean()

        return action

    def val(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        # TODO: Currently not using func; remove later
        obs = torch.from_numpy(obs).float().to(self.device)
        act = torch.from_numpy(act).float().to(self.device)
        values = torch.sum((self.q(obs) * act), dim=-1, keepdim=True)
        values = values.detach().cpu().numpy()
        self._clean()

        return values

    def forward(self, obs: np.ndarray) -> np.ndarray:
        # TODO: Currently not using func; remove later
        obs = torch.from_numpy(obs).float().to(self.device)
        action = self.act(obs)
        value = self.val(obs, action)
        self._clean()

        return action, value

    def update_trg(self):
        self.q.update_trg(alpha=self.polyak)

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
                 update_interval: int = 10000,
                 train_interval: int = 1,
                 num_epochs: int = 1,
                 buffer_cls: Type[ReplayBuffer] = ExperienceReplay,
                 buffer_size: int = int(1e6),
                 buffer_kwargs: dict = None,
                 batch_size: int = 32,
                 explore: bool = True,
                 loss_func: Callable = loss_func,
                 save_interval: int = int(1e5),
                 eps: float = 0.1,
                 decay_step: int = int(1e6),
                 gamma: float = 0.99,
                 log_interval: int = int(1e3),
                 train_after: int = int(1e3),
                 update_after: int = int(1e3),
                 **kwargs):
        if loss_func is None:
            self.loss_func = loss_func

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
        self.update_interval = update_interval
        self.train_interval = train_interval
        self.save_interval = save_interval
        self.log_interval = log_interval

        # Set hyperparams
        self.update_after = update_after
        self.train_after = train_after
        self.batch_size = batch_size
        self.explore = explore
        self.gamma = gamma
        self.eps = LinearDecay(start=1, end=eps, decay_step=decay_step)

        # Set loss function
        self.loss_func = loss_func

        # Set logger
        self.logger = kwargs.get('logger')
        if self.logger:
            self.log_dir = self.logger.log_dir

    def act(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) in (1, 3):
            obs = np.expand_dims(obs, axis=0)

        if self.explore and np.random.random() < self.eps(self.curr_step):
            action = np.random.randint(self.model.action_shape[0], size=())
        else:
            action = self.model.act(obs)

        return action

    def step(self, s, a, r, d, s_):
        self.collect(s, a, r, d, s_)
        info = {'Values/EPS': self.eps(self.curr_step)}

        if (self.curr_step % self.train_interval == 0 and
                self.curr_step > self.train_after):
            info_t = self.train()
            info.update(info_t)
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
            self.model.q.step(loss)
        # FIXME: bug; Remaining buffer obs
        self.model._clean()

        info = {
            'Loss/Q_network': loss.item(),
        }

        return info

    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        if self.model.discrete:
            a = np.eye(self.model.action_shape[0])[a]
        self.buffer.push(s, a, r, d, s_)
