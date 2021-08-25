#!/usr/bin/env python3

import numpy as np
from marlenv.wrappers import make_snake

from rl2 import TEST_DATA_DIR
from rl2.agents.gail import GAILAgent, discriminator
from rl2.agents.ppo import PPOModel, PPOAgent
from rl2.data_utils import ExpertTrajectory
from rl2.workers import MaxStepWorker

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
    expert_trajs = ExpertTrajectory(num_episodes=8, one_hot=one_hot)
    expert_trajs.load_pickle(f'{TEST_DATA_DIR}/PPOAgent_trajs.pickle')

    disc = discriminator(obs_shape, ac_shape)
    agent = GAILAgent(
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
