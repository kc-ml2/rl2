from typing import Callable
import os
import torch
import torch.nn.functional as F
import numpy as np

from rl2.agents.base import Agent
from rl2.models.base import TorchModel, BranchModel
from rl2.agents.utils import general_advantage_estimation
from rl2.buffers import ReplayBuffer, TemporalMemory


def loss_func(
        data, model,
        hidden=None, clip_param=0.1, vf_coef=0.5, ent_coef=0.001
):
    obs, old_acs, dones, _, old_vals, old_nlps, advs = data
    obs, old_acs, dones, old_vals, old_nlps, advs = map(
        lambda x: torch.from_numpy(x).float().to(model.device),
        [obs, old_acs, dones, old_vals, old_nlps, advs]
    )
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

    def __init__(
            self,
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
            **kwargs
    ):
        super().__init__(observation_shape, action_shape, **kwargs)
        if hasattr(encoder, 'output_shape'):
            encoded_dim = encoder.output_shape
        # Handles only the 1-dim action space
        self.encoded_dim = encoded_dim
        self.recurrent = recurrent
        self.policy = BranchModel(
            observation_shape, action_shape,
            encoded_dim=encoded_dim,
            discrete=discrete,
            deterministic=deterministic,
            flatten=flatten,
            reorder=reorder,
            recurrent=self.recurrent,
            **kwargs
        )

        self.value = BranchModel(
            observation_shape, (1,),
            encoder=self.policy.encoder,
            encoded_dim=encoded_dim,
            discrete=True,
            deterministic=True,
            flatten=flatten,
            reorder=reorder,
            recurrent=self.recurrent,
            **kwargs
        )
        self.init_params(self.policy)
        self.init_params(self.value)

    @TorchModel.sharedbranch
    def forward(self, obs, **kwargs) -> torch.tensor:
        # The observation should be shaped differently rather
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

        return action_dist, value_dist

    def act(self, obs: np.ndarray, get_log_prob=False) -> np.ndarray:
        action_dist, hidden = self._infer_from_numpy(self.policy, obs)
        action = action_dist.sample().squeeze()
        if get_log_prob:
            log_prob = action_dist.log_prob(action)
            log_prob = log_prob.detach().cpu().numpy()
        action = action.detach().cpu().numpy()

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

        info = {}
        if self.recurrent:
            info['hidden'] = hidden

        return value

    def save(self, save_dir):
        torch.save(
            self.state_dict(),
            os.path.join(save_dir, type(self).__name__ + '.pt')
        )
        print(f'model saved in {save_dir}')

    def load(self, load_dir):
        ckpt = torch.load(load_dir, map_location=self.device)
        self.load_state_dict(ckpt)


class PPOAgent(Agent):
    def __init__(
            self,
            model: TorchModel,
            train_interval: int = 0,
            eval_interval: int = 0,
            num_epochs: int = 1,
            num_envs: int = 1,
            buffer_cls: ReplayBuffer = TemporalMemory,
            buffer_kwargs: dict = {},
            batch_size: int = 128,
            loss_func: Callable = loss_func,
            update_after: int = 1,
            val_coef: float = 0.5,
            gamma: float = 0.99,
            lamda: float = 0.95,
    ):
        super().__init__(
            model,
            train_interval=train_interval,
            eval_interval=eval_interval,
            num_epochs=num_epochs,
            buffer_cls=buffer_cls,
            buffer_kwargs=buffer_kwargs,
            num_envs=num_envs,
        )
        self.obs = None
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_after = update_after

        self.value = None
        self.nlp = None
        self.loss_func = loss_func
        self.val_coef = val_coef
        self.lamda = lamda

    def act(self, obs):
        action, info = self.model.act(obs, get_log_prob=True)
        self.nlp = -info['log_prob']
        self.value = self.model.val(obs)

        # TODO: should recurrent relavent calls be handled in base class?
        if self.model.recurrent:
            self.hidden = info['hidden']

        return action

    def train_at(self, curr_step):
        return curr_step % self.train_interval == 0

    def step(self, state, action, reward, done, next_state):
        self.curr_step += 1
        self.collect(state, action, reward, self.done, self.value, self.nlp)
        self.done = done

        if self.model.recurrent:
            self.model._update_hidden(done, self.hidden)

        info = {}
        if self.train_at(self.curr_step):
            value = self.model.val(next_state)
            advs = general_advantage_estimation(
                self.buffer.to_dict(), value, done, self.gamma, self.lamda
            )
            info = self.train(advs)
            self.buffer.reset()

            if self.model.recurrent:
                self.prev_hidden = self.model.hidden

        return info

    def train(self, advs):
        losses = []
        for _ in range(self.num_epochs):
            num_minibatches = (
                    self.buffer.curr_size * self.num_envs // self.batch_size
            )
            self.buffer.shuffle()
            for mb_idx in range(num_minibatches):
                batch_data, sub_idx = self.sample_batch_data(advs)

                if self.model.recurrent:
                    env_idx = sub_idx.reshape(self.buffer.max_size, -1)[0]
                    hidden = tuple([ph[:, env_idx] for ph in self.prev_hidden])
                else:
                    hidden = None

                loss = self.loss_func(batch_data, self.model, hidden=hidden)
                self.model.policy.step(loss, retain_graph=True)
                self.model.value.step(loss)
                losses.append(loss.item())

        info = {
            'Loss/All': sum(losses) / (len(losses) + 1e-8)
        }
        # print(info)

        return info

    def sample_batch_data(self, advs):
        batch_data = self.buffer.sample(
            self.batch_size, return_idx=True,
            recurrent=self.model.recurrent)
        idx, sub_idx = batch_data[-1]
        batch_data = (
            *batch_data[:-1],
            np.expand_dims(advs[idx, sub_idx], axis=1)
        )
        return batch_data, sub_idx

    def collect(self, state, action, reward, done, value, nlp):
        self.buffer.push(
            state=state,
            action=action,
            reward=reward,
            done=self.done,
            value=self.value,
            nlp=self.nlp
        )
