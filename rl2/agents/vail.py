import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rl2.agents.utils import general_advantage_estimation
from rl2.data_utils import flatten_concat
from rl2.distributions import DiagGaussianDist
from rl2.agents.gail import GAILAgent, loss_fn

from rl2.models.base import BranchModel

DUAL_LR = 5e-5
PPO_LR = None
DISC_LR = 1e-4


def loss_fn(logits, labels, kld, beta):
    information_constrain = 0.5
    dual_lr = DUAL_LR

    bottleneck_loss = kld - information_constrain
    beta = max(0., beta + dual_lr * bottleneck_loss.detach())

    disc_loss = F.binary_cross_entropy_with_logits(logits, labels)
    disc_loss = disc_loss + beta * bottleneck_loss

    return disc_loss, beta


class VDB(nn.Module):
    def __init__(self, observation_shape, action_shape, latent_size):
        super().__init__()
        encoded_dim = 256

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # input_shape = np.prod(observation_shape) + action_shape[0]

        input_shape = (
            *observation_shape[:-1],
            observation_shape[-1] + action_shape[0]
        )
        print(input_shape)
        # in_channels = observation_shape[-1] + action_shape[0]
        # enc = nn.Sequential(
        #     nn.Conv2d(in_channels, 64, 3),
        #     nn.MaxPool2d(),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3),
        #     nn.MaxPool2d(),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3),
        #     nn.MaxPool2d(),
        #     nn.ReLU(),
        # )
        # enc = nn.Sequential(
        #     nn.Linear(input_shape, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, encoded_dim),
        # )

        self.encoder = BranchModel(
            observation_shape=input_shape,
            action_shape=(latent_size,),
            # encoder=enc,
            # encoded_dim=encoded_dim,
            deterministic=False,
            discrete=False,
            lr=DISC_LR,
        )

        denc = nn.Sequential(
            nn.Linear(latent_size, encoded_dim),
            nn.ReLU(),
            nn.Linear(encoded_dim, encoded_dim),
        )

        self.discriminator = BranchModel(
            (latent_size,),
            (1,),
            encoder=denc,
            encoded_dim=encoded_dim,
            discrete=False,
            deterministic=True,
            lr=DISC_LR,
        )
        self.dist = None

    def step(self, loss):
        e_opt = self.encoder.optimizer
        d_opt = self.discriminator.optimizer

        e_opt.zero_grad()
        d_opt.zero_grad()

        loss.backward()

        e_opt.step()
        d_opt.step()

    def forward(self, x):
        self.dist = self.encoder(x)
        # TODO: look at self.dist.rsample() <-> geunyang sample gradient ggeungeo
        latent = self.dist.mean + self.dist.stddev * torch.randn_like(
            self.dist.stddev)
        logits = self.discriminator(latent)

        return logits


class VAILAgent:
    """
    1. different discriminator; add bottleneck
    2. regularize loss with kl
    """

    def __init__(
            self,
            rlagent,
            model,
            batch_size,
            expert_trajs,
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
        self.flatten = self.expert_trajs.flatten
        self.one_hot = one_hot
        self.beta = 0.

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
        self.collect(state, action, reward, agent.done, agent.value,
                     agent.nlp)
        agent.done = done
        info = {}
        if agent.train_at(agent.curr_step):
            # self.buffer.shuffle()
            info_ = self.train()
            # print(info_)
            if self.flatten:
                adversary = flatten_concat(
                    self.buffer.state,
                    self.buffer.action,
                    self.one_hot,
                ).to(self.model.device)
            else:
                adversary = []
                for st, ac in zip(self.buffer.state, self.buffer.action):
                    for i_st, i_ac in zip(st, ac):
                        ac_embed = self.one_hot[i_ac]
                        st_ac = np.concatenate((i_st, ac_embed), -1)
                        adversary.append(st_ac)

                adversary = np.asarray(adversary)
                adversary = torch.from_numpy(adversary).to(
                    self.model.device)
            self.buffer.reward = self.disc_reward(adversary)

            # TODO: only PPO, currently
            value = agent.model.val(next_state)
            advs = general_advantage_estimation(
                self.buffer.to_dict(), value, done, agent.gamma, agent.lamda
            )
            agent.train(advs)

    def train(self):
        info = {}
        losses = []
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for expert_batch, expert_labels in self.expert_traj_loader:
                buffer_batch = self.buffer.sample(
                    self.batch_size,
                    return_idx=True
                )
                if self.flatten:
                    buffer_batch = flatten_concat(
                        buffer_batch[0],
                        buffer_batch[1],
                        self.one_hot,
                    ).to(self.model.device)
                else:
                    action_embed = np.asarray(
                        [self.one_hot[ac[0]].numpy() for ac in buffer_batch[1]]
                    )
                    buffer_batch = np.concatenate([buffer_batch[0], action_embed], -1)
                    buffer_batch = torch.from_numpy(buffer_batch).to(self.model.device)

                buffer_labels = torch.zeros(len(buffer_batch)).to(
                    self.model.device
                )

                batch = torch.cat([expert_batch, buffer_batch])
                labels = torch.cat([expert_labels, buffer_labels]).unsqueeze(1)

                logits = self.model(batch).mean

                kld = self.model.dist.kl(
                    DiagGaussianDist(0., 1.)
                ).mean()
                disc_loss, self.beta = self.loss_fn(logits, labels, kld,
                                                    self.beta)

                info_ = self.model.step(disc_loss)

                epoch_losses.append(disc_loss)
            losses.append(sum(epoch_losses) / len(epoch_losses))

        return losses

    def act(self, obs):
        actions = self.rlagent.act(obs)
        return actions

    def collect(self, *args):
        self.rlagent.collect(*args)
