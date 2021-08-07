import logging
import os
import pickle
import signal
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Tuple

import gym
import numpy as np

from rl2.agents.base import Agent
from rl2.ctx import var
from rl2.workers.infotrack import InfoTracker

logger = logging.getLogger(__name__)


class RolloutWorker:
    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
            render_interval: int = 0,
            save_interval: int = 0,
            log_interval: int = 0,
            save_erange: Tuple[int, int] = None,  # range of episodes
            render_mode: str = '',
    ):
        config = var.get()
        self.base_log_dir = config['log_dir']

        self.env = env
        self.agent = agent
        self.render_interval = render_interval
        self.render_mode = render_mode
        self.save_interval = save_interval

        self.curr_episode = 0
        self.num_steps = 0

        self.scores = deque(maxlen=100)
        self.episode_length = deque(maxlen=100)
        self.episode_score = 0
        self.episode_steps = 0

        self.obs = env.reset()

        self.save_erange = save_erange
        self.curr_trajectory = []
        self.trajectories = []

        self.log_interval = log_interval

        self.saving_model = True

    def run(self):
        raise NotImplementedError

    def __enter__(self):
        self.start_dt = time.clock()

        if self.saving_model:
            signal.signal(signal.SIGINT, lambda sig, frame: self.save_model())

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_dt = time.clock()

        if self.saving_model:
            try:
                self.save_model()
            except Exception as e:
                logger.warning(f'saving model failed, {e}')

        logging.info(f'Time elapsed {self.end_dt - self.start_dt}.')
        logging.info(f'Ran from {self.start_dt} to {self.end_dt}.')

    @property
    def running_time(self):
        return self.end_dt - self.start_dt

    def set_mode(self, train=True):
        # for torch currently
        if train:
            self.train_mode = True
            # self.agent.model.train()
        else:
            self.train_mode = False
            # self.agent.model.eval()

    def as_saving(self, tensorboard=True, saved_model=True):
        self.saving_model = saved_model
        self.tensorboard = tensorboard

        return self

    def default_save_dir(self):
        return f"""{type(self).__name__}_{datetime.now().strftime('%Y%m%d%H%M%S')}"""

    def ensure_save_dir(self, name):
        save_dir = os.path.join(
            self.base_log_dir,
            self.default_save_dir(),
            name
        )

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        return save_dir

    def save_model(self, save_dir=None):
        if save_dir is None:
            save_dir = self.ensure_save_dir('models')

        save_dir = os.path.join(
            save_dir,
            f'ckpt/{int(self.num_steps / 1000)}k'
        )
        Path(save_dir).mkdir(parents=True)
        self.agent.model.save(save_dir)

        logging.info(f'saved model to {save_dir}')

        return save_dir

    def save_expert_data(self, save_dir=None):
        if save_dir is None:
            save_dir = self.ensure_save_dir('expert_data')

        filename = f'{type(self.agent).__name__}_trajs.pickle'
        save_dir = os.path.join(save_dir, filename)

        with open(save_dir, 'wb') as fp:
            pickle.dump(self.trajectories, fp)

        logging.info(f'saved expert trajectories to {save_dir}')

        return save_dir

    def rollout(self):
        action = self.agent.act(self.obs)
        if self.in_erange():
            # S_t, A_t
            self.curr_trajectory.append((self.obs, action))

        if hasattr(action, 'shape'):
            if len(action.shape) == 2 and action.shape[0] == 1:
                action = action.squeeze(0)
        obs, reward, done, env_info = self.env.step(action)

        # if self.train_mode:
        agent_info = self.agent.step(self.obs, action, reward, done, obs)
        if env_info:
            if isinstance(env_info, dict):
                env_info = {**env_info, **agent_info}
            elif isinstance(env_info, list) or isinstance(env_info, tuple):
                env_info = {**agent_info}
        else:
            env_info = {**agent_info}

        self.num_steps += self.agent.num_envs
        self.episode_score = self.episode_score + np.array(reward)
        self.episode_steps = self.episode_steps + np.ones_like(done, np.int)

        done = np.asarray(done)

        if done.size > 1:
            self.curr_episode += sum(done)
            for idx in np.where(done)[0]:
                self.scores.append(self.episode_score[idx])
                self.episode_score[idx] = 0.
                self.episode_length.append(self.episode_steps[idx])
                self.episode_steps[idx] = 0.
        else:
            if done:
                # episode changes from below
                obs = self.env.reset()
                self.scores.append(self.episode_score)
                self.episode_score = 0.
                self.episode_length.append(self.episode_steps)
                self.episode_steps = 0

                if self.in_erange():
                    self.trajectories.append(self.curr_trajectory)
                    self.curr_trajectory = []

                self.curr_episode += 1

        self.obs = obs
        results = None

        return done, env_info, results

    def in_erange(self):
        if self.save_erange is None:
            return False
        return self.save_erange[0] <= self.curr_episode < self.save_erange[1]


class MaxStepWorker(RolloutWorker):
    """
    do rollout until max steps given
    """

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
            max_steps: int = 10,
            log_interval: int = 1,
            render_interval: int = 128,
            save_interval: int = 128,
            save_erange: Tuple[int, int] = None,
            render_mode: str = None,
            logger=None,
            log_vars=[],
            log_smooth_vars=[],
    ):
        super().__init__(
            env=env,
            agent=agent,
            log_interval=log_interval,
            render_interval=render_interval,
            save_interval=save_interval,
            save_erange=save_erange,
            render_mode=render_mode,
        )
        self.max_steps = int(max_steps)
        log_vars += ['Counts/num_steps', 'Counts/num_episodes']
        log_smooth_vars += ['Episodic/rews_avg', 'Episodic/ep_length',
                            'episode_rewards']
        self.info = InfoTracker(log_vars, smoothing_keys=log_smooth_vars,
                                smoothing_const=100)

        self.logger = logger
        self.store_image = False
        self.start_log_image = False
        self.time_to_log_image = False

    def run(self):
        steps_per_env = (self.max_steps // self.agent.num_envs) + 1
        for step in range(steps_per_env):
            done, info, results = self.rollout()
            self.info.update(info)
            self.worker_log(done)

            if self.save_at():
                self.save_model()

    def save_at(self):
        return (self.save_interval > 0) and (
                (self.num_steps % self.save_interval) < self.agent.num_envs)

    # def save_model(self):
    #     if hasattr(self, 'logger'):
    #         save_dir = getattr(self.logger, 'log_dir')
    #     else:
    #         save_dir = os.getcwd()
    #
    #     self.save_model(save_dir)

    def worker_log(self, done):
        # Save rendered image as gif
        # if self.render_interval > 0:
        #     if (self.num_steps % self.render_interval) < self.agent.num_envs:
        #         self.time_to_log_image = True
        #
        #     cond = done if np.asarray(done).size == 1 else done[0]
        #     if cond:
        #         if self.time_to_log_image:
        #             self.time_to_log_image = False
        #             self.start_log_image = True
        #         elif self.start_log_image:
        #             self.start_log_image = False
        #             self.store_image = True
        #
        #     if self.start_log_image:
        #         image = self.env.render(self.render_mode)
        #         self.logger.store_rgb(image)
        #     elif self.store_image:
        #         self.logger.video_summary(
        #             tag='playback', step=self.num_steps
        #         )
        #         self.store_image = False
        # Log info
        if self.num_steps % self.log_interval < self.agent.num_envs:
            info_r = {
                'Counts/num_steps': self.num_steps,
                'Counts/num_episodes': self.curr_episode,
                'Episodic/rews_avg': np.mean(list(self.scores)),
                'Episodic/ep_length': np.mean(list(self.episode_length))
            }
            self.info.update(info_r)
            self.logger.scalar_summary(self.info, self.num_steps)
        # # self.info.update(info)


class EpisodicWorker(RolloutWorker):
    """
    do rollout until max episodes given
    might be useful at inference time or when training episodically
    """

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
            max_episodes: int = 10,
            log_interval: int = 1,
            render_interval: int = 128,
            save_interval: int = 128,
            save_erange: Tuple[int, int] = None,
            render_mode: str = None,
    ):
        super().__init__(
            env=env,
            agent=agent,
            log_interval=log_interval,
            render_interval=render_interval,
            save_interval=save_interval,
            save_erange=save_erange,
            render_mode=render_mode,
        )
        self.max_episodes = int(max_episodes)
        self.info = {}
        self.store_image = False
        self.start_log_image = False

    def run(self):
        while self.curr_episode < self.max_episodes:
            # print(f'running episode {self.curr_episode}...')
            prev_num_ep = self.curr_episode
            done, info, results = self.rollout()

            self.worker_log(done, prev_num_ep)

    def worker_log(self, done, prev_num_ep):
        if self.render_interval > 0 and self.start_log_image:
            image = self.env.render(self.render_mode)
            self.logger.store_rgb(image)
        log_cond = done if np.asarray(done).size == 1 else any(done)
        if log_cond:
            if self.start_log_image:
                print('save video')
                self.logger.video_summary(tag='playback',
                                          step=self.num_steps)
                self.start_log_image = False
            if self.render_interval > 0:
                if (prev_num_ep // self.render_interval !=
                        self.curr_episode // self.render_interval):
                    self.start_log_image = True
            info_r = {
                'Counts/num_steps': self.num_steps,
                'Counts/num_episodes': self.curr_episode,
                'Episodic/rews_avg': np.mean(list(self.scores)),
                'Episodic/ep_length': np.mean(list(self.episode_length))
            }
            self.info.update(info_r)
            # self.info.update(info)
            if (prev_num_ep // self.log_interval
                    != self.curr_episode // self.log_interval):
                self.logger.scalar_summary(self.info, self.num_steps)
