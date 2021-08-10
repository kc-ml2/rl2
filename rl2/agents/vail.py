import numpy as np
import torch
from torch.nn import functional as F

from rl2.data_utils import flatten_concat
from rl2.distributions import DiagGaussianDist
from rl2.agents.gail import GAILAgent, loss_fn

from rl2.models.base import BranchModel


def loss_fn(logits, labels, kld, beta):
    information_constrain = 0.5
    dual_lr = 1e-5

    bottleneck_loss = kld - information_constrain
    beta = max(0., beta + dual_lr * bottleneck_loss)

    disc_loss = F.binary_cross_entropy_with_logits(logits, labels)
    disc_loss = disc_loss + beta * bottleneck_loss

    return disc_loss, beta


class VDB(BranchModel):
    def __init__(self, observation_shape, action_shape, latent_size):
        super().__init__(observation_shape, action_shape)
        self.lr = int(1e-4)
        input_shape = np.prod(observation_shape) + action_shape[0]
        self.encoder = BranchModel(
            observation_shape=(input_shape,),
            action_shape=(latent_size,),
            deterministic=False,
            discrete=False,
            flatten=True,
        )
        self.discriminator = BranchModel(
            (latent_size,),
            (1,),
            discrete=False,
            deterministic=True,
        )
        self.dist = None

    def forward(self, x):
        self.dist = self.encoder(x)
        # TODO: look at self.dist.rsample() <-> geunyang sample gradient ggeungeo
        latent = self.dist.mean + self.dist.stddev * torch.randn_like(
            self.dist.stddev)
        logits = self.discriminator(latent)

        return logits


class VAILAgent(GAILAgent):
    """
    1. different discriminator; add bottleneck
    2. regularize loss with kl
    """

    def __init__(
            self,
            model,
            discriminator,
            expert_trajs,
            one_hot,
            num_envs,
            disc_batch_size,
            loss_fn=loss_fn,
            **kwargs,
    ):
        GAILAgent.__init__(
            self, model=model, num_envs=num_envs, one_hot=one_hot,
            expert_trajs=expert_trajs, discriminator=discriminator,
            disc_loss_fn=loss_fn, disc_batch_size=disc_batch_size, **kwargs
        )
        self.loss_fn = loss_fn
        self.beta = 1e-5

    def train_discriminator(self):
        info = {}
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

                batch = torch.cat([expert_batch, buffer_batch])
                labels = torch.cat([expert_labels, buffer_labels]).unsqueeze(1)

                logits = self.discriminator(batch).mean

                kld = self.discriminator.dist.kl(
                    DiagGaussianDist(0., 1.)
                ).mean()
                disc_loss, self.beta = self.loss_fn(logits, labels, kld,
                                                    self.beta)

                info_ = self.discriminator.step(disc_loss)

                epoch_losses.append(disc_loss)
            losses.append(sum(epoch_losses) / len(epoch_losses))

        return losses