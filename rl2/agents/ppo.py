from typing import Callable
import torch
import torch.nn.functional as F
import numpy as np
from collections.abc import Iterable

from rl2.agents.base import Agent
from rl2.models.torch.base import TorchModel, BranchModel
from rl2.agents.utils import general_advantage_estimation
from rl2.buffers import ReplayBuffer, TemporalMemory


def loss_func(data, model,
              hidden=None, clip_param=0.1, vf_coef=0.5, ent_coef=0.001):

    obs, old_acs, dones, _, old_vals, old_nlps, advs = data
    obs, old_acs, dones, old_vals, old_nlps, advs = map(
        lambda x: torch.from_numpy(x).float().to(model.device),
        [obs, old_acs, dones, old_vals, old_nlps, advs])
    val_targets = old_vals + advs

    # Infer from model
    ac_dist, val_dist = model(obs, hidden=hidden, mask=dones)
    nlps = -ac_dist.log_prob(old_acs.squeeze()).unsqueeze(-1)
    ent = ac_dist.entropy().mean()
    vals = val_dist.mean

    vals_clipped = (old_vals + torch.clamp(vals - old_vals,
                                           -clip_param, clip_param))
    vf_loss_clipped = F.mse_loss(vals_clipped, val_targets)
    vf_loss = F.mse_loss(vals, val_targets)
    vf_loss = torch.max(vf_loss, vf_loss_clipped).mean()

    # Standardize advantage
    advs = (advs - advs.mean()) / (advs.std() + 1e-7)

    ratio = torch.exp(old_nlps - nlps)
    pg_loss1 = -advs * ratio

    ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    pg_loss2 = -advs * ratio
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    loss = pg_loss - ent_coef * ent + vf_coef * vf_loss

    return loss


class PPOModel(TorchModel):
    """
    predefined model
    (same one as original paper)
    """
    def __init__(self,
                 observation_shape,
                 action_shape,
                 encoder: torch.nn.Module = None,
                 encoded_dim: int = 64,
                 optimizer='torch.optim.Adam',
                 lr=1e-4,
                 discrete: bool = True,
                 deterministic: bool = False,
                 flatten: bool = False,
                 reorder: bool = False,
                 recurrent: bool = False,
                 **kwargs):
        super().__init__(observation_shape, action_shape, **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape
        # Handles only the 1-dim action space
        self.encoded_dim = encoded_dim
        self.recurrent = recurrent
        self.policy = BranchModel(observation_shape, action_shape,
                                  encoded_dim=encoded_dim,
                                  discrete=discrete,
                                  deterministic=deterministic,
                                  flatten=flatten,
                                  reorder=reorder,
                                  recurrent=self.recurrent,
                                  **kwargs)

        self.value = BranchModel(observation_shape, (1,),
                                 encoded_dim=encoded_dim,
                                 discrete=False,
                                 deterministic=True,
                                 flatten=flatten,
                                 reorder=reorder,
                                 recurrent=self.recurrent,
                                 **kwargs)
        self.init_params(self.policy)
        self.init_params(self.value)

    def _clean(self):
        # TODO: this function should be moved to base class
        for mod in self.children():
            if isinstance(mod, BranchModel):
                mod.reset_encoder_memory()

    def _update_hidden(self, dones, new_hidden):
        # TODO: this function should be moved to base class
        hidden = new_hidden[0]
        cellstate = new_hidden[1]
        done_idx = np.where(np.asarray(dones) == 1)[0]
        hidden[0, done_idx, :] = torch.zeros(
            len(done_idx), self.encoded_dim).to(self.device)
        cellstate[0, done_idx, :] = torch.zeros(
            len(done_idx), self.encoded_dim).to(self.device)
        self.hidden = (hidden, cellstate)

    def _init_hidden(self, dones):
        # TODO: this function should be moved to base class
        if not isinstance(dones, Iterable):
            dones = [dones]
        H = torch.zeros(1, len(dones), self.encoded_dim).to(self.device)
        C = torch.zeros(1, len(dones), self.encoded_dim).to(self.device)
        self.hidden = (H, C)

    def forward(self, obs, **kwargs) -> torch.tensor:
        # TODO: The observation should be shaped differently rather
        # the model is currently using recurrent embedding or not.
        # It is assumed that the minibatch will have a full sequence so
        # each forward call should never have to receive hidden state
        # at this point
        obs = obs.to(self.device)
        action_dist = self.policy(obs, **kwargs)
        value_dist = self.value(obs, **kwargs)
        if self.recurrent:
            action_dist = action_dist[0]
            value_dist = value_dist[0]

        # TODO: All calls to _clean should be handled using a decorator
        self._clean()
        return action_dist, value_dist

    def _infer_from_numpy(self, net, obs):
        # TODO: this function should be moved to base class
        obs = torch.from_numpy(obs).float().to(self.device)
        # if len(observation.shape) == len(self.observation_shape):
        #     observation = observation.unsqueeze(0)
        hidden = None
        with torch.no_grad():
            if self.recurrent:
                dist, hidden = net(obs.unsqueeze(0), hidden=self.hidden)
            else:
                dist = net(obs)

        return dist, hidden

    def act(self, obs: np.ndarray, get_log_prob=False) -> np.ndarray:
        action_dist, hidden = self._infer_from_numpy(self.policy, obs)
        action = action_dist.sample().squeeze()
        if get_log_prob:
            log_prob = action_dist.log_prob(action)
            log_prob = log_prob.detach().cpu().numpy()
        action = action.detach().cpu().numpy()

        self._clean()
        info = {}
        if get_log_prob:
            info['log_prob'] = log_prob
        if self.recurrent:
            info['hidden'] = hidden

        return action, info

    def val(self, obs: np.ndarray) -> np.ndarray:
        value_dist, hidden = self._infer_from_numpy(self.value, obs)
        value = value_dist.mean.squeeze()
        value = value.detach().cpu().numpy()

        self._clean()
        info = {}
        if self.recurrent:
            info['hidden'] = hidden
        return value

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass


class PPOAgent(Agent):
    def __init__(self,
                 model: TorchModel,
                 train_interval: int = 128,
                 num_epochs: int = 1,
                 n_env=1,
                 buffer_cls: ReplayBuffer = TemporalMemory,
                 buffer_kwargs: dict = None,
                 batch_size: int = 128,
                 val_coef: float = 0.5,
                 action_low: np.array = None,
                 action_high: np.ndarray = None,
                 loss_func: Callable = loss_func,
                 save_interval: int = int(1e5),
                 gamma: float = 0.99,
                 lamda: float = 0.95,
                 log_interval: int = int(1e3),
                 **kwargs):
        # self.buffer = ReplayBuffer()
        super().__init__(model, train_interval, num_epochs,
                         buffer_cls, buffer_kwargs, **kwargs)

        # TODO: some of these can be moved to base class
        self.obs = None
        self.n_env = n_env
        if self.n_env == 1:
            self.done = False
        else:
            self.done = [False] * n_env
        self.gamma = gamma
        self.batch_size = batch_size

        self.value = None
        self.nlp = None
        self.loss_func = loss_func
        self.val_coef = val_coef
        self.lamda = lamda

        # TODO: For rnn encoding. Should this be moved to base class?
        self.model._init_hidden(self.done)
        self.hidden = self.model.hidden
        self.pre_hidden = self.hidden

    def act(self, obs):
        action, info = self.model.act(obs, get_log_prob=True)
        self.nlp = -info['log_prob']
        self.value = self.model.val(obs)

        # TODO: should recurrent relavent calls be handled in base class?
        if self.model.recurrent:
            self.hidden = info['hidden']

        return action

    def step(self, s, a, r, d, s_):
        self.curr_step += 1
        self.collect(s, a, r, self.done, self.value, self.nlp)
        if self.curr_step % self.train_interval == 0:
            value = self.model.val(s_)
            advs = general_advantage_estimation(self.buffer.to_dict(),
                                                value, d,
                                                self.gamma, self.lamda)
            self.train(advs)
            self.buffer.reset()
        self.done = d

        # TODO: should recurrent method be handled in base class?
        if self.model.recurrent:
            self.model._update_hidden(d, self.hidden)
            self.pre_hidden = self.model.hidden

        info = dict()
        return info

    def train(self, advs, **kwargs):
        for _ in range(self.num_epochs):
            batch_data = self.buffer.sample(self.batch_size, return_idx=True,
                                            recurrent=self.model.recurrent)
            idx, sub_idx = batch_data[-1]
            batch_data = (*batch_data[:-1],
                          np.expand_dims(advs[idx, sub_idx], axis=1))
            if self.model.recurrent:
                env_idx = sub_idx.reshape(self.buffer.max_size, -1)[0]
                hidden = tuple([ph[:, env_idx] for ph in self.pre_hidden])
            else:
                hidden = None
            loss = self.loss_func(batch_data, self.model, hidden=hidden)
            self.model.policy.step(loss, retain_graph=True)
            self.model.value.step(loss)

    def collect(self, *args):
        self.buffer.push(*args)
