import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable, Type
from rl2.agents.base import Agent
from rl2.buffers.base import ExperienceReplay, ReplayBuffer
from rl2.models.torch.base import BranchModel, TorchModel
from rl2.agents.utils import LinearDecay


def loss_func(data, model, hidden=None, **kwargs):
    s, a, r, d, s_ = tuple(
        map(lambda x: torch.from_numpy(x).float().to(model.device), data)
    )
    done_mask = copy.deepcopy(d)
    if model.recurrent:
        done_mask[:, 0] = 1
    done_mask_ = d
    with torch.no_grad():
        if model.recurrent:
            q_next_trg = model.q.forward_trg(s_, mask=done_mask_)[0].mean
        else:
            q_next_trg = model.q.forward_trg(s_, mask=done_mask_).mean
        if model.double:
            q_next = model(s_, mask=done_mask).mean
            a_ = torch.argmax(q_next, dim=-1, keepdim=True)
            v_trg = torch.gather(q_next_trg, dim=-1, index=a_)
        else:
            v_trg = torch.max(q_next_trg, dim=-1, keepdim=True).values
        bellman_trg = r + kwargs['gamma'] * v_trg * (1 - d.reshape(-1, 1))

    q = torch.sum((model(s, mask=done_mask).mean * a), dim=-1, keepdim=True)
    loss = F.smooth_l1_loss(q, bellman_trg)

    return loss


class DQNModel(TorchModel):
    """
    predefined model
    (same one as original paper)
    """
    def __init__(self,
                 observation_shape,
                 action_shape,
                 encoder: torch.nn.Module = None,
                 encoded_dim: int = 64,
                 double: bool = False,  # Double DQN if True
                 optimizer: str = 'torch.optim.RMSprop',
                 lr: float = 1e-4,
                 grad_clip: float = 1,
                 polyak: float = float(0),
                 discrete: bool = True,
                 flatten: bool = False,
                 reorder: bool = False,
                 recurrent: bool = False,
                 **kwargs):
        super().__init__(observation_shape, action_shape, **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape
        self.encoded_dim = encoded_dim
        self.recurrent = recurrent
        self.is_save = kwargs.get('is_save', False)
        self.discrete = discrete
        self.double = double
        self.rews_ep = 0

        # Set hyperparams
        self.polyak = polyak

        # Set default RMSprop optim_args
        if optimizer == 'torch.optim.RMSprop':
            kwargs.setdefault('optim_args', {'alpha': 0.95,
                                             'eps': 0.00001,
                                             'momentum': 0.0,
                                             'centered': True})

        self.q = BranchModel(observation_shape,
                             action_shape,
                             encoded_dim=encoded_dim,
                             optimizer=optimizer,
                             lr=lr,
                             grad_clip=grad_clip,
                             make_target=True,
                             discrete=discrete,
                             deterministic=True,
                             recurrent=recurrent,
                             reorder=reorder,
                             flatten=flatten,
                             head_depth=1,
                             **kwargs)
        self.init_params(self.q)

    def forward(self, obs: torch.Tensor, **kwargs) -> np.ndarray:
        # TODO: Currently not using func; remove later
        obs = obs.to(self.device)
        value_dist = self.q(obs, **kwargs)
        if self.recurrent:
            value_dist = value_dist[0]

        return value_dist

    def act(self, obs: np.array) -> np.ndarray:
        value_dist, hidden = self._infer_from_numpy(self.q, obs)
        values = value_dist.mean
        action = torch.argmax(values)
        action = action.detach().cpu().numpy()

        info = {}
        if self.recurrent:
            info['hidden'] = hidden

        return action, info

    def val(self, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
        # TODO: Currently not using func; remove later
        value_dist, hidden = self._infer_from_numpy(self.q, obs)
        values = value_dist.mean
        values = values.detach().cpu().numpy()
        values = (values * act).sum(axis=-1, keepdims=True)

        info = {}
        if self.recurrent:
            info['hidden'] = hidden

        return values

    def update_trg(self):
        self.q.update_trg(alpha=self.polyak)

    def save(self, save_dir):
        torch.save(self.state_dict(),
                   os.path.join(save_dir, type(self).__name__ + '.pt'))
        print(f'model saved in {save_dir}')

    def load(self, load_dir):
        ckpt = torch.load(load_dir, map_location=self.device)
        self.load_state_dict(ckpt)


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
        self.eps = LinearDecay(start=0.9, end=eps, decay_step=decay_step)
        if isinstance(eps, LinearDecay):
            self.eps = eps

        # Set loss function
        self.loss_func = loss_func

        # For recurrent intermediate representation
        self.done = False
        self.model._init_hidden(self.done)
        self.hidden = self.model.hidden
        self.pre_hidden = self.hidden

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, info = self.model.act(obs)
        if self.explore and np.random.random() < self.eps(self.curr_step):
            action = np.random.randint(self.model.action_shape[0], size=())

        if self.model.recurrent:
            self.hidden = info['hidden']

        return action

    def step(self, s, a, r, d, s_):
        self.curr_step += 1
        self.collect(s, a, r, d, s_)
        self.done = d

        if self.model.recurrent:
            self.model._update_hidden(d, self.hidden)

        info = {'Values/EPS': self.eps(self.curr_step)}

        if (self.curr_step % self.train_interval == 0 and
                self.curr_step > self.train_after):
            info_t = self.train()
            info.update(info_t)
        if (self.curr_step % self.update_interval == 0 and
                self.curr_step > self.update_after):
            self.model.update_trg()

        return info

    def train(self):
        for _ in range(self.num_epochs):
            if self.model.recurrent:
                contiguous = 8
            else:
                contiguous = 1
            batch = self.buffer.sample(self.batch_size, contiguous=contiguous)
            loss = self.loss_func(batch, self.model, gamma=self.gamma)
            self.model.q.step(loss)

        info = {
            'Loss/Q_network': loss.item(),
        }

        return info

    def collect(self, s, a, r, d, s_):
        if self.model.discrete:
            a = np.eye(self.model.action_shape[0])[a]
        self.buffer.push(s, a, r, d, s_)
