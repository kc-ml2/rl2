from typing import Callable
import torch
import torch.nn.functional as F
import numpy as np

from rl2.agents.base import Agent
from rl2.models.torch.base import TorchModel, BranchModel
from rl2.agents.utils import general_advantage_estimation
from rl2.buffers import ReplayBuffer, TemporalMemory


def loss_func(data, model, clip_param=0.1, vf_coef=0.5, ent_coef=0.001):
    obs, old_acs, _, _, old_vals, old_nlps, advs = data
    obs, old_acs, old_vals, old_nlps, advs = map(
        lambda x: torch.from_numpy(x).float().to(model.device),
        [obs, old_acs, old_vals, old_nlps, advs])
    val_targets = old_vals + advs
    # Infer from model
    ac_dist, val_dist = model(obs)
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
                 **kwargs):
        super().__init__(observation_shape, action_shape, **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape
        # Handles only the 1-dim action space
        self.policy = BranchModel(observation_shape, action_shape,
                                  encoded_dim=encoded_dim,
                                  discrete=discrete,
                                  deterministic=deterministic,
                                  flatten=flatten,
                                  reorder=reorder,
                                  **kwargs)

        self.value = BranchModel(observation_shape, (1,),
                                 encoded_dim=encoded_dim,
                                 discrete=False,
                                 deterministic=True,
                                 flatten=flatten,
                                 reorder=reorder,
                                 **kwargs)
        self.init_params(self.policy)
        self.init_params(self.value)

    def _clean(self):
        for mod in self.children():
            if isinstance(mod, BranchModel):
                mod.reset_encoder_memory()

    def forward(self, obs) -> torch.tensor:
        obs = obs.to(self.device)
        action_dist = self.policy(obs)
        value_dist = self.value(obs)

        self._clean()
        return action_dist, value_dist

    def act(self, obs: np.ndarray, get_log_prob=False) -> np.ndarray:
        obs = torch.from_numpy(obs).float().to(self.device)
        action_dist = self.policy(obs)
        action = action_dist.sample().squeeze()
        if get_log_prob:
            log_prob = action_dist.log_prob(action)
            log_prob = log_prob.detach().cpu().numpy()
        action = action.detach().cpu().numpy()

        self._clean()
        if get_log_prob:
            return action, log_prob
        return action

    def val(self, obs: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(obs).float().to(self.device)
        value = self.value(obs).mean.squeeze()
        value = value.detach().cpu().numpy()

        self._clean()
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
        self.model = model

        self.obs = None
        self.done = False
        self.value = None
        self.nlp = None
        self.loss_func = loss_func
        self.val_coef = val_coef
        self.gamma = gamma
        self.lamda = lamda
        self.batch_size = batch_size

    def act(self, obs):
        action, log_prob = self.model.act(obs, get_log_prob=True)
        self.value = self.model.val(obs)
        self.nlp = -log_prob

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

        info = dict()
        return info

    def train(self, advs, **kwargs):
        for _ in range(self.num_epochs):
            batch_data = self.buffer.sample(self.batch_size, return_idx=True)
            idx = batch_data[-1]
            batch_data = (*batch_data[:-1], advs[idx])
            loss = self.loss_func(batch_data, self.model)
            self.model.policy.step(loss, retain_graph=True)
            self.model.value.step(loss)

    def collect(self, *args):
        self.buffer.push(*args)
