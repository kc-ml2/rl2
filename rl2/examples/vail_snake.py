#!/usr/bin/env python3

import numpy as np
from marlenv.wrappers import make_snake

from rl2 import TEST_DATA_DIR
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.agents.vail import VDB, VAILAgent
from rl2.data_utils import FlatExpertTrajectory
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

env, obs_shape, ac_shape, props = make_snake(num_envs=NUM_ENVS, num_snakes=1, width=7, height=7, vision_range=5, frame_stack=2)


if __name__ == '__main__':
    model = PPOModel(obs_shape, ac_shape)
    rlagent = PPOAgent(
        model=model,
        train_interval=TRAIN_INTERVAL,
        num_envs=NUM_ENVS,
        batch_size=512,
        buffer_kwargs={
            'size': TRAIN_INTERVAL
        }
    )

    one_hot = np.eye(env.action_space[0].n)
    expert_trajs = FlatExpertTrajectory(num_episodes=8, one_hot=one_hot)
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
    # env, obs_shape, ac_shape, props = make_snake(
    #     num_envs=64,
    #     num_snakes=1,
    #     width=7,
    #     height=7,
    #     vision_range=5,
    #     frame_stack=2
    # )
    #
    # one_hot = np.eye(env.action_space[0].n)
    # expert_trajs = FlatExpertTrajectory(num_episodes=8, one_hot=one_hot)
    # expert_trajs.load_pickle(f'{TEST_DATA_DIR}/PPOAgent_trajs.pickle')
    #
    # model = PPOModel(obs_shape, ac_shape)
    # vdb = VDB(obs_shape, ac_shape, latent_size=5)
    #
    # agent = VAILAgent(
    #     model=model,
    #     discriminator=vdb,
    #     expert_trajs=expert_trajs,
    #     num_epochs=4,
    #     train_interval=TRAIN_INTERVAL,
    #     num_envs=props['num_envs'],
    #     buffer_kwargs={
    #         'size': TRAIN_INTERVAL,
    #     },
    #     one_hot=one_hot,
    #     disc_batch_size=BATCH_SIZE,
    #     loss_fn=loss_fn
    # )
    #
    # worker = MaxStepWorker(
    #     env,
    #     agent,
    #     max_steps=1024 ** 2,
    #     render_interval=0,
    #     log_interval=1024,
    #     save_interval=0
    # )
    # with worker.as_saving(tensorboard=False, saved_model=False):
    #     worker.run()
