import os
from abc import abstractmethod

import torch
from torch import nn

from rl2.models.torch.base import PolicyGradientModel


class ActorCriticModel(PolicyGradientModel):
    """
    actor critic is also an interface, but serves as vanilla actor critic(original paper)
    """
    def __init__(self, input_shape, encoder, actor, critic, **kwargs):
        super().__init__(input_shape, **kwargs)
        self.encoder = encoder
        self.actor = actor
        self.critic = critic

        # there can be multiple optimizers in subclasses
        # order doesn't matter
        self.optimizer = torch.optim.Adam(
            [encoder.parameters(), actor.parameters(), critic.parameters()]
            # self.parameters()
        )

        self.max_grad = None

    @abstractmethod
    def forward(self, x):
        """
        for training
        :param x:
        :return: distributions
        """
        ir = self.encoder(x)
        ac_dist = self.actor(ir)
        val_dist = self.critic(ir)

        return ac_dist, val_dist

    @abstractmethod
    def infer(self, x):
        """
        :return: distributions
        """
        ir = self.encoder(x)
        ac_dist = self.actor(ir)

        return ac_dist

    @abstractmethod
    def step(self, loss):
        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad)
        nn.utils.clip_grad_norm(self.encoder.parameters(), self.max_grad)
        nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad)

        self.optimizer.step()

    @abstractmethod
    def save(self, save_dir):
        # torch.save(os.path.join(save_path, 'actor_critic.pt'), self.state_dict())
        torch.save(os.path.join(save_dir, 'encoder.pt'))
        torch.save(os.path.join(save_dir, 'actor.pt'))
        torch.save(os.path.join(save_dir, 'critic.pt'))

    @abstractmethod
    def load(self, save_dir):
        self.encoder = torch.load(os.path.join(save_dir, 'encoder.pt'))
        self.actor = torch.load(os.path.join(save_dir, 'actor.pt'))
        self.critic = torch.load(os.path.join(save_dir, 'critic.pt'))
