#!/usr/bin/env python3
import numpy as np
import torch
from marlenv.wrappers import make_snake
from torch.nn import functional as F

from rl2 import TEST_DATA_DIR
from rl2.agents.ppo import PPOModel
from rl2.distributions import DiagGaussianDist
from rl2.examples.gail_snake_v2 import GAILAgent, FlatExpertTrajectory, \
    flatten_concat, disc_loss_fn
from rl2.models.base import BranchModel
from rl2.workers import MaxStepWorker


def loss_fn(logits, labels, kld, beta):
    information_constrain = 0.5
    dual_lr = 1e-5

    bottleneck_loss = kld - information_constrain
    max(0., beta + dual_lr * bottleneck_loss)

    disc_loss = F.binary_cross_entropy_with_logits(logits, labels)
    disc_loss = disc_loss + beta * bottleneck_loss

    return disc_loss, beta


class VDB(BranchModel):
    def __init__(self, observation_shape, action_shape, latent_size):
        super().__init__(observation_shape, action_shape)
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
            loss_fn,
            disc_batch_size,
            **kwargs,
    ):
        GAILAgent.__init__(
            self, model=model, num_envs=num_envs, one_hot=one_hot,
            expert_trajs=expert_trajs, discriminator=discriminator,
            disc_loss_fn=disc_loss_fn, disc_batch_size=disc_batch_size, **kwargs
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
                    DiagGaussianDist(0., 1.)).mean()
                disc_loss, self.beta = self.loss_fn(logits, labels, kld,
                                                    self.beta)

                info_ = self.discriminator.step(disc_loss)

                epoch_losses.append(disc_loss)
            losses.append(sum(epoch_losses) / len(epoch_losses))

        return losses


if __name__ == '__main__':
    TRAIN_INTERVAL = 128
    BATCH_SIZE = 512

    env, obs_shape, ac_shape, props = make_snake(num_envs=64, num_snakes=1,
                                                 vision_range=5,
                                                 frame_stack=2)

    # how can we integrate num_envs?
    # set it in worker, lazy init else where...

    # config = var.get('config')
    # list vs element for single agent... marlenv...
    one_hot = np.eye(env.action_space[0].n)
    expert_trajs = FlatExpertTrajectory(num_episodes=8, one_hot=one_hot)
    expert_trajs.load_pickle(f'{TEST_DATA_DIR}/PPOAgent_trajs.pickle')

    # x1 = torch.randn(5, 11, 11, 16)
    # x2 = torch.randn(1941)

    model = PPOModel(obs_shape, ac_shape)
    vdb = VDB(obs_shape, ac_shape, latent_size=5)

    # y1, aa = model(x1)
    # y2 = vdb(x2)
    #
    # make_dot(aa.mean, params=dict(model.named_parameters())).render('graph1', format='png')
    # make_dot(y2.mean, params=dict(vdb.named_parameters())).render('graph2', format='png')

    # oo = e.observation_space.sample()
    # oo = torch.FloatTensor(oo)
    # aa = e.action_space.sample()
    # writer = SummaryWriter('./summary')
    # writer.add_graph(model, oo)
    # vv = model(oo)
    # writer.add_graph(vdb, vv)
    # writer.close()

    agent = VAILAgent(
        model=model,
        discriminator=vdb,
        expert_trajs=expert_trajs,
        num_epochs=4,
        train_interval=TRAIN_INTERVAL,
        num_envs=props['num_envs'],
        buffer_kwargs={
            'size': TRAIN_INTERVAL,
        },
        one_hot=one_hot,
        disc_batch_size=BATCH_SIZE,
        loss_fn=loss_fn
    )

    worker = MaxStepWorker(
        env,
        agent,
        max_steps=1024 ** 2,
        render_interval=0,
        log_interval=1024,
        save_interval=0
    )
    with worker.as_saving(tensorboard=False, saved_model=False):
        worker.run()
