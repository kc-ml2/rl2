import os
import numpy as np
from pathlib import Path
from collections import deque
from rl2.agents.base import Agent


class RolloutWorker:
    """
    workers mimics the intuitive loop btw agent and env

    if worker is to serve as some entrypoint for an app,
    worker might need context so everything is under control of workers

    rl2's base unit is a step(1 interaction per se)
    """

    def __init__(
            self,
            env,
            n_env,
            agent: Agent,
            training=False,
            render=False,
            render_interval=1000,
            is_save=False,
            save_interval=int(1e6),
            # render_mode: str ='human',
            **kwargs
    ):
        self.env = env
        self.n_env = n_env
        self.agent = agent
        self.training = training

        self.render = render
        if self.render:
            self.render_interval = render_interval
            self.render_mode = kwargs.get('render_mode', 'rgb_array')

        self.is_save = is_save
        if self.is_save:
            self.save_interval = save_interval

        self.num_episodes = 0
        self.num_steps = 0
        self.scores = deque(maxlen=100)
        self.ep_length = deque(maxlen=100)
        self.episode_score = 0
        self.ep_steps = 0

        self.obs = env.reset()

    # def register(self, agent):
    #     self.agent.add(agent)

    def run(self):
        raise NotImplementedError

    def save(self, save_dir):
        save_dir = os.path.join(save_dir, f'ckpt/{int(self.num_steps/1000)}k')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.agent.model.save(save_dir)

    def rollout(self):
        ac = self.agent.act(self.obs)
        if len(ac.shape) == 2 and ac.shape[0] == 1:
            ac = ac.squeeze(0)
        obs, rew, done, info = self.env.step(ac)
        if self.training:
            info_a = self.agent.step(self.obs, ac, rew, done, obs)
            if info:
                if isinstance(info, dict):
                    info = {**info, **info_a}
                elif isinstance(info, list) or isinstance(info, tuple):
                    info = {**info_a}
                    # for env_i, info_i in enumerate(info):
                    #     info['env_{}'.format(env_i)] = info_i
            else:
                info = {**info_a}
            # task_list = self.agent.dispatch()
            # if len(task_list) > 0:
            #     results = {
            #         bound_method.__name__: bound_method()
            #         for bound_method in task_list}
        self.num_steps += self.n_env
        self.episode_score = self.episode_score + np.array(rew)
        self.ep_steps = self.ep_steps + np.ones_like(done, np.int)
        done = np.asarray(done)
        if done.size > 1:
            self.num_episodes += sum(done)
            for d_i in np.where(done)[0]:
                self.scores.append(self.episode_score[d_i])
                self.episode_score[d_i] = 0.
                self.ep_length.append(self.ep_steps[d_i])
                self.ep_steps[d_i] = 0.
        else:
            if done:
                self.num_episodes += 1
                obs = self.env.reset()
                self.scores.append(self.episode_score)
                self.episode_score = 0.
                self.ep_length.append(self.ep_steps)
                self.ep_steps = 0
        self.obs = obs
        results = None

        return done, info, results


class MaxStepWorker(RolloutWorker):
    """
    do rollout until max steps given
    """

    def __init__(self, env, n_env, agent,
                 max_steps=1000, logger=None,
                 log_interval=5000, **kwargs):
        super().__init__(env, n_env, agent, **kwargs)
        self.max_steps = int(max_steps)
        self.log_interval = int(log_interval)
        self.logger = logger
        self.info = {}
        self.store_image = False
        self.start_log_image = False
        self.time_to_log_image = False

    def run(self):
        for step in range(self.max_steps // self.n_env + 1):
            done, info, results = self.rollout()

            # Save rendered image as gif
            if self.render:
                if self.num_steps % self.render_interval < self.n_env:
                    self.time_to_log_image = True

                cond = done if np.asarray(done).size == 1 else done[0]
                if cond:
                    if self.time_to_log_image:
                        self.time_to_log_image = False
                        self.start_log_image = True
                    elif self.start_log_image:
                        self.start_log_image = False
                        self.store_image = True

                if self.start_log_image:
                    image = self.env.render(self.render_mode)
                    self.logger.store_rgb(image)
                elif self.store_image:
                    self.logger.video_summary(tag='playback',
                                              step=self.num_steps)
                    self.store_image = False

            # Log info
            if self.num_steps % self.log_interval < self.n_env:
                info_r = {
                    'Counts/num_steps': self.num_steps,
                    'Counts/num_episodes': self.num_episodes,
                    'Episodic/rews_avg': np.mean(list(self.scores)),
                    'Episodic/ep_length': np.mean(list(self.ep_length))
                }
                self.info.update(info_r)
                # self.info.update(info)
                self.logger.scalar_summary(self.info, self.num_steps)

            # Save model
            if (self.is_save and
               self.num_steps % self.save_interval < self.n_env):
                if hasattr(self, 'logger'):
                    save_dir = getattr(self.logger, 'log_dir')
                else:
                    save_dir = os.getcwd()
                self.save(save_dir)


class EpisodicWorker(RolloutWorker):
    """
    do rollout until max episodes given
    might be useful at inference time or when training episodically
    """

    def __init__(self, env, n_env, agent,
                 max_episodes: int = 10,
                 log_interval: int = 1,
                 logger=None,
                 **kwargs):
        super().__init__(env, n_env, agent, **kwargs)
        self.max_episodes = int(max_episodes)
        self.log_interval = int(log_interval)
        self.logger = logger
        self.info = {}
        self.store_image = False
        self.start_log_image = False

    def run(self):
        while self.num_episodes < self.max_episodes:
            prev_num_ep = self.num_episodes
            done, info, results = self.rollout()

            if self.render and self.start_log_image:
                image = self.env.render(self.render_mode)
                self.logger.store_rgb(image)

            log_cond = done if np.asarray(done).size == 1 else any(done)
            if log_cond:
                if self.start_log_image:
                    print('save video')
                    self.logger.video_summary(tag='playback',
                                              step=self.num_steps)
                    self.start_log_image = False
                if self.render:
                    if (prev_num_ep // self.render_interval !=
                       self.num_episodes // self.render_interval):
                        self.start_log_image = True
                info_r = {
                    'Counts/num_steps': self.num_steps,
                    'Counts/num_episodes': self.num_episodes,
                    'Episodic/rews_avg': np.mean(list(self.scores)),
                    'Episodic/ep_length': np.mean(list(self.ep_length))
                }
                self.info.update(info_r)
                # self.info.update(info)
                if (prev_num_ep // self.log_interval
                   != self.num_episodes // self.log_interval):
                    self.logger.scalar_summary(self.info, self.num_steps)
