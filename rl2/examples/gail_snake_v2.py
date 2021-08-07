#!/usr/bin/env python3

import numpy as np
import torch
from marlenv.wrappers import make_snake
from torch.utils.data import DataLoader
from torch.nn import functional as F
from rl2 import TEST_DATA_DIR
from rl2.agents import PPOAgent
from rl2.agents.mixins import AdversarialImitationMixin
from rl2.agents.ppo import PPOModel
from rl2.agents.utils import general_advantage_estimation
from rl2.data_utils import FlatExpertTrajectory, flatten_concat
from rl2.models.base import BranchModel
from rl2.workers import MaxStepWorker


e, o, a, p = make_snake(num_envs=64, num_snakes=1, width=7, height=7, vision_range=5, frame_stack=2)


def disc_loss_fn(logits, labels):
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    # loss = -torch.log(labels -  torch.sigmoid(logits) + 1e-8).sum()

    return loss


discriminatior = BranchModel(
    (np.prod(o) + a[0],),
    (1,),
    discrete=False,
    deterministic=True,
    flatten=True,
    lr=0.0001,
)


class GAILAgent(AdversarialImitationMixin, PPOAgent):
    def __init__(
            self,
            model,
            discriminator,
            expert_trajs: FlatExpertTrajectory,
            one_hot,
            num_envs,
            disc_batch_size,
            disc_loss_fn=disc_loss_fn,
            **kwargs,
    ):
        PPOAgent.__init__(self, model=model, num_envs=num_envs, batch_size=512, **kwargs)
        self.disc_loss_fn = disc_loss_fn
        self.discriminator = discriminator
        self.expert_trajs = expert_trajs
        self.disc_batch_size = disc_batch_size
        self.disc_epochs = 1
        self.expert_traj_loader = DataLoader(
            expert_trajs,
            batch_size=disc_batch_size,
            drop_last=True,
        )
        self.one_hot = one_hot
        self.outer_shape = (self.train_interval, self.num_envs)

    def discrimination_reward(self, adversary):
        with torch.no_grad():
            logits = self.discriminator(adversary).mean.squeeze()
            cost = -torch.log(1 - torch.sigmoid(logits) + 1e-8).view(self.train_interval, self.num_envs)
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
            print(info_)
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
        losses = []
        for epoch in range(self.disc_epochs):
            epoch_losses = []
            for expert_batch, expert_labels in self.expert_traj_loader:
                buffer_batch = self.buffer.sample(
                    self.disc_batch_size,
                    return_idx=True
                )

                buffer_batch = flatten_concat(
                    buffer_batch[0],
                    buffer_batch[1],
                    self.one_hot,
                ).to(self.discriminator.device)

                buffer_labels = torch.zeros(len(buffer_batch)).to(
                    self.discriminator.device
                )
                # print(expert_batch.shape, buffer_batch.shape)
                # print(expert_labels.shape, buffer_labels.shape)
                batch = torch.cat([expert_batch, buffer_batch])
                labels = torch.cat([expert_labels, buffer_labels])
                prob = self.discriminator(batch)
                logits = prob.mean.squeeze()

                disc_loss = self.disc_loss_fn(logits, labels)
                info = self.discriminator.step(disc_loss)

                epoch_losses.append(disc_loss)
            # losses.append(sum(epoch_losses)/len(epoch_losses))
        return losses


if __name__ == '__main__':
    TRAIN_INTERVAL = 128
    BATCH_SIZE = 1024
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
        disc_batch_size=BATCH_SIZE,
        one_hot=one_hot
    )

    worker = MaxStepWorker(e, agent, max_steps=1024 ** 2, render_interval=0,
                           log_interval=1024, save_interval=0)
    with worker.as_saving(tensorboard=False, saved_model=False):
        worker.run()
