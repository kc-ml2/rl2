import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rl2.agents.base import Agent
from rl2.agents.utils import general_advantage_estimation
from rl2.data_utils import FlatExpertTrajectory, flatten_concat
from rl2.models.base import BranchModel


def loss_fn(logits, labels):
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    return loss


def discriminator(obs_shape, action_shape):
    model = BranchModel(
        (np.prod(obs_shape) + action_shape[0],),
        (1,),
        discrete=False,
        deterministic=True,
        flatten=True,
        lr=0.0001,
    )

    return model


class GAILAgent(Agent):
    """
    model : discriminator
    batch_size : discriminator's batch size
    loss_fn : discriminator's loss function
    num_epochs : discriminator's number of epochs
    """

    def __init__(
            self,
            rlagent,
            model,
            batch_size,
            expert_trajs: FlatExpertTrajectory,
            one_hot,
            loss_fn=loss_fn,
            num_epochs=1,
    ):
        self.rlagent = rlagent
        self.buffer = rlagent.buffer
        self.buffer_shape = (self.buffer.max_size, self.buffer.num_envs)
        self.num_envs = rlagent.num_envs
        self.model = model
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.expert_trajs = expert_trajs
        self.num_epochs = num_epochs
        self.expert_traj_loader = DataLoader(
            expert_trajs,
            batch_size=batch_size,
            drop_last=True,
        )
        self.one_hot = one_hot

    def disc_reward(self, adversary):
        with torch.no_grad():
            logits = self.model(adversary).mean.squeeze()
            cost = -torch.log(1 - torch.sigmoid(logits) + 1e-8).view(
                *self.buffer_shape)
            cost = cost.cpu().numpy().tolist()

        return cost

    def step(self, state, action, reward, done, next_state):
        agent = self.rlagent
        agent.curr_step += 1
        self.collect(state, action, reward, agent.done, agent.value, agent.nlp)
        agent.done = done
        info = {}
        if agent.train_at(agent.curr_step):
            # self.buffer.shuffle()
            info_ = self.train()
            print(info_)
            adversary = flatten_concat(
                self.buffer.state,
                self.buffer.action,
                self.one_hot,
            ).to(self.model.device)

            self.buffer.reward = self.disc_reward(adversary)

            # TODO: only PPO, currently
            value = agent.model.val(next_state)
            advs = general_advantage_estimation(
                self.buffer.to_dict(), value, done, agent.gamma, agent.lamda
            )
            agent.train(advs)
            # value = self.model.val(next_state)
            # advs = general_advantage_estimation(
            #     self.buffer.to_dict(), value, done, self.gamma, self.lamda
            # )
            #
            # info_ = self.train(advs)
            #
            # self.buffer.reset()
            #
            # if self.model.recurrent:
            #     self.prev_hidden = self.model.hidden

        return info

    def train(self):
        losses = []
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for expert_batch, expert_labels in self.expert_traj_loader:
                buffer_batch = self.buffer.sample(
                    self.batch_size,
                    return_idx=True
                )

                buffer_batch = flatten_concat(
                    buffer_batch[0],
                    buffer_batch[1],
                    self.one_hot,
                ).to(self.model.device)

                buffer_labels = torch.zeros(len(buffer_batch)).to(
                    self.model.device
                )
                batch = torch.cat([expert_batch, buffer_batch])
                labels = torch.cat([expert_labels, buffer_labels])
                logits = self.model(batch).mean.squeeze()

                disc_loss = self.loss_fn(logits, labels)
                self.model.step(disc_loss)

                epoch_losses.append(disc_loss)

        return losses

    def act(self, obs):
        actions = self.rlagent.act(obs)
        return actions

    def collect(self, *args):
        self.rlagent.collect(*args)