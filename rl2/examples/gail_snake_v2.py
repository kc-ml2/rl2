import pickle
from typing import List, Tuple

import numpy as np
import torch
from marlenv.wrappers import make_snake
from torch.utils.data import Dataset, DataLoader

from rl2 import TEST_DATA_DIR
from rl2.agents import PPOAgent
from rl2.agents.base import Agent
from rl2.agents.ppo import PPOModel
from rl2.agents.utils import general_advantage_estimation
from rl2.models.base import BranchModel
from rl2.workers import MaxStepWorker

TRAIN_INTERVAL = 128
BATCH_SIZE = 16

e, o, a, p = make_snake(num_envs=3, num_snakes=1, vision_range=5, frame_stack=2)


def disc_loss_fn(logits, labels):
    loss = -torch.log(labels - torch.sigmoid(logits) + 1e-8).sum()

    return loss


class FlatExpertTrajectory(Dataset):
    def __init__(
            self,
            # expects list of trajectory of (state, action)
            data: List[List[Tuple[np.ndarray, np.ndarray]]] = None,
            num_episodes: int = None,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            one_hot: np.ndarray = None,
    ):
        self.num_episodes = num_episodes
        self.device = device
        self.one_hot = one_hot
        self.data_ = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @property
    def data(self):
        return self.data_

    @data.setter
    def data(self, val):
        if self.num_episodes is not None:
            episodes = np.random.randint(0, len(val), size=self.num_episodes)
            val = [val[i] for i in episodes]

        flat_data = self._flatten(val)
        data = torch.from_numpy(flat_data)
        self.data_ = data.to(self.device)
        self.labels = torch.ones(len(self.data)).to(self.device)

    def _flatten(self, data):
        flat_data = []
        for traj in data:
            flat_traj = []
            for state, action in traj:
                action_embedding = self.one_hot[action]
                state_action = [
                    np.asarray(state.flatten()),
                    action_embedding,
                ]
                state_action = np.concatenate(state_action)
                flat_traj.append(state_action)
            flat_data.append(state_action)
        flat_data = np.asarray(flat_data).astype(np.float32)

        return flat_data

    def load_pickle(self, data_dir):
        with open(data_dir, 'rb') as fp:
            self.data = pickle.load(fp)
            # type check?


discriminatior = BranchModel(
    (np.prod(o) + a[0],),
    (1,),
    discrete=False,
    deterministic=True,
    flatten=True
)


class AdversarialImitationMixin:
    def discrimination_reward(self):
        raise NotImplementedError

    def train_discriminator(self):
        raise NotImplementedError


def flatten_concat(states, actions, one_hot):
    if not isinstance(states, np.ndarray) or not isinstance(actions,
                                                            np.ndarray):
        states = np.asarray(states)
        actions = np.asarray(actions)
        rows, cols = states.shape[0], states.shape[1]
        states = states.reshape(rows * cols, -1)
        actions = actions.reshape(rows * cols, -1)

    states = [state.flatten().astype(np.float32) for state in states]
    actions = [one_hot[int(action[0])] for action in actions]
    ret = np.asarray([np.concatenate([i, j]) for i, j in zip(states, actions)])
    ret = torch.from_numpy(ret).float()

    return ret


class GAILAgent(AdversarialImitationMixin, PPOAgent):
    def __init__(
            self,
            model,
            discriminator,
            expert_trajs: FlatExpertTrajectory,
            one_hot,
            num_envs,
            disc_loss_fn=disc_loss_fn,
            **kwargs,
    ):
        PPOAgent.__init__(self, model=model, num_envs=num_envs, **kwargs)
        self.disc_loss_fn = disc_loss_fn
        self.discriminator = discriminator
        self.expert_trajs = expert_trajs
        self.disc_batch_size = BATCH_SIZE
        self.disc_epochs = 10
        self.expert_traj_loader = DataLoader(
            expert_trajs,
            batch_size=self.disc_batch_size
        )
        self.one_hot = one_hot
        self.outer_shape = (self.train_interval, self.num_envs)

    def discrimination_reward(self, adversary):
        with torch.no_grad():
            logits = self.discriminator(adversary).mean.squeeze()
            error = 1 - torch.sigmoid(logits)
            cost = -torch.log(error + 1e-8).view(128, 3)
            cost = cost.cpu().numpy().tolist()

        return cost

    def step(self, state, action, reward, done, next_state):
        self.curr_step += 1
        self.collect(state, action, reward, self.done, self.value, self.nlp)
        self.done = done
        info = {}
        if self.train_at(self.curr_step):
            self.buffer.shuffle()
            info_ = self.train_discriminator()

            adversary = flatten_concat(
                self.buffer.state,
                self.buffer.action,
                self.one_hot,
            ).to(self.discriminator.device)

            self.buffer.reward = self.discrimination_reward(adversary)

            value = self.model.val(next_state)
            advs = general_advantage_estimation(
                self.buffer.to_dict(), value, done, self.gamma, self.lamda
            )

            info_ = self.train(advs)

            self.buffer.reset()

            if self.model.recurrent:
                self.prev_hidden = self.model.hidden

        return info

    def train_discriminator(self):
        for epoch in range(self.disc_epochs):
            for expert_batch, expert_labels in self.expert_traj_loader:
                buffer_batch = self.buffer.sample(
                    self.disc_batch_size,
                    return_idx=True
                )

                buffer_batch = flatten_concat(
                    buffer_batch[0],
                    buffer_batch[1],
                    one_hot,
                ).to(self.discriminator.device)

                buffer_labels = torch.zeros(len(buffer_batch)).to(
                    self.discriminator.device
                )
                batch = torch.cat([expert_batch, buffer_batch])
                labels = torch.cat([expert_labels, buffer_labels])
                prob = self.discriminator(batch)
                logits = prob.mean.squeeze()

                disc_loss = self.disc_loss_fn(logits, labels, self.outer_shape)
                info = self.discriminator.step(disc_loss)

        return info


if __name__ == '__main__':
    one_hot = np.eye(e.action_space[0].n)
    expert_trajs = FlatExpertTrajectory(num_episodes=8, one_hot=one_hot)
    expert_trajs.load_pickle(f'{TEST_DATA_DIR}/PPOAgent_trajs.pickle')
    model = PPOModel(o, a)
    agent = GAILAgent(
        model=model,
        discriminator=discriminatior,
        expert_trajs=expert_trajs,
        train_interval=TRAIN_INTERVAL,
        num_envs=p['num_envs'],
        buffer_kwargs={
            'size': TRAIN_INTERVAL,
        },
        one_hot=one_hot
    )

    worker = MaxStepWorker(e, agent, max_steps=1024, render_interval=16)
    worker.set_mode(train=True)
    with worker.as_saving(tensorboard=False, saved_model=False):
        worker.run()