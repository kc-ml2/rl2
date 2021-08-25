#!/usr/bin/env python3

import torch
from marlenv.wrappers import make_snake

from rl2 import TEST_DATA_DIR
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.agents.vail import VDB, VAILAgent
from rl2.data_utils import ExpertTrajectory
from rl2.workers import MaxStepWorker

# def visualize_graph():
#     from torchviz import make_dot
#     x1 = torch.randn(5, 11, 11, 16)
#     x2 = torch.randn(1941)
#     y1, y2 = model(x1)
#     y3 = vdb(x2)
#
#     make_dot(
#         y2.mean, params=dict(model.named_parameters())
#     ).render('graph1', format='png')
#     make_dot(
#         y3.mean, params=dict(vdb.named_parameters())
#     ).render('graph2', format='png')

TRAIN_INTERVAL = 128
BATCH_SIZE = 1024
NUM_ENVS = 64

env, obs_shape, ac_shape, props = make_snake(num_envs=NUM_ENVS, num_snakes=1,
                                             width=7, height=7, vision_range=5,
                                             frame_stack=2)

from torch import nn


class ActionOneHot2d(nn.Module):
    def __init__(self, num_classes, data_shape):
        super(ActionOneHot2d, self).__init__()
        assert len(data_shape) == 2, 'must be 2d image shape'
        self.eval()
        self.num_classes = num_classes
        self.data_shape = data_shape
        embeddings = []
        # change this to torch.tile for torch version > 1.6
        for i in range(num_classes):
            embedding = torch.zeros(num_classes, *data_shape)
            embedding[i] = 1.
            embeddings.append(embedding)

        # TODO: shape management
        self.embeddings = torch.stack(embeddings).transpose(1, -1)
        self.shape = self.embeddings.shape

    def __getitem__(self, idx):
        return self.embeddings[idx]

    def forward(self, x):
        return self.embeddings[x]


if __name__ == '__main__':
    model = PPOModel(obs_shape, ac_shape, lr=5e-5)
    rlagent = PPOAgent(
        model=model,
        train_interval=TRAIN_INTERVAL,
        num_envs=NUM_ENVS,
        batch_size=512,
        buffer_kwargs={
            'size': TRAIN_INTERVAL
        }
    )

    # one_hot = np.eye(env.action_space[0].n)
    one_hot = ActionOneHot2d(ac_shape[0], obs_shape[:-1])
    expert_trajs = ExpertTrajectory(num_episodes=8, one_hot=one_hot)
    # expert_trajs = ExpertTrajectory1d(num_episodes=8, one_hot=one_hot)
    expert_trajs.load_pickle(f'{TEST_DATA_DIR}/PPOAgent_trajs.pickle')

    disc = VDB(obs_shape, ac_shape, latent_size=128)
    agent = VAILAgent(
        rlagent=rlagent,
        model=disc,
        batch_size=BATCH_SIZE,
        expert_trajs=expert_trajs,
        one_hot=one_hot,
    )

    worker = MaxStepWorker(
        env, agent,
        max_steps=int(5e6),
        render_interval=0,
        log_interval=int(1e4),
        save_interval=0
    )
    with worker.as_saving(tensorboard=True, saved_model=False):
        worker.run()
