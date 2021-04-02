import os
import numpy as np
from pathlib import Path
from collections import deque
from collections.abc import Iterable
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
        if self.render and self.start_log_image:
            render_result = self.env.render(self.render_mode)
            info['image'] = render_result

        self.num_steps += self.n_env
        self.episode_score = self.episode_score + np.array(rew)
        done = np.asarray(done)
        self.ep_steps = self.ep_steps + np.ones(shape=done.shape)
        if isinstance(done, Iterable):
            if any(done):
                self.num_episodes += sum(done)
                for d_i, d in enumerate(done):
                    if d:
                        self.scores.append(self.episode_score[d_i])
                        self.episode_score[d_i] = 0.
                        self.ep_length.append(self.ep_steps[d_i])
                        self.ep_steps[d_i] = 0.
        else:
            if done:  # do sth about ven env
                self.num_episodes += 1
                obs = self.env.reset()
                self.scores.append(self.episode_score)
                self.episode_score = 0.
        # Update next obs
        self.obs = obs
        info = {**info, 'rew': rew}
        results = None

        return done, info, results


class MaxStepWorker(RolloutWorker):
    """
    do rollout until max steps given
    """

    def __init__(self, env, n_env, agent,
                 max_steps: int, logger=None,
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
            if (self.num_steps % self.render_interval < self.n_env
               and self.render):
                self.time_to_log_image = True

            if self.render:
                if self.start_log_image:
                    self.logger.store_rgb(info['image'])
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
                info.pop('rew')
                if 'image' in info.keys():
                    info.pop('image')
                self.info.update(info)
                self.logger.scalar_summary(self.info, self.num_steps)

            # Save model
            if (self.is_save and
               self.num_steps % self.save_interval < self.n_env):
                if hasattr(self, 'logger'):
                    save_dir = getattr(self.logger, 'log_dir')
                self.save(save_dir)

            cond = done if type(done) == bool else done[0]
            if cond:
                if self.time_to_log_image:
                    self.time_to_log_image = False
                    self.start_log_image = True
                elif self.start_log_image:
                    self.start_log_image = False
                    self.store_image = True



class EpisodicWorker(RolloutWorker):
    """
    do rollout until max episodes given
    might be useful at inference time or when training episodically
    """

    def __init__(self, env, n_env, agent,
                 max_episodes: int = 10,
                 max_steps_per_ep: int = int(1e4),
                 log_interval: int = 1000,
                 logger=None,
                 **kwargs):
        super().__init__(env, n_env, agent, **kwargs)
        self.max_episodes = int(max_episodes)
        self.max_steps_per_ep = int(max_steps_per_ep)
        self.log_interval = int(log_interval)
        self.num_steps_ep = 0
        self.scores = deque(maxlen=100)
        self.logger = logger

    def run(self):
        for episode in range(self.max_episodes):
            for num_steps_ep in range(self.max_steps_per_ep):
                done, info, results = self.rollout()
                self.rews += info['rew']
                if done:
                    self.scores.append(self.rews)
                    avg_score = np.mean(list(self.scores))
                    info_r = {
                        'Counts/num_steps': self.num_steps,
                        'Counts/num_episodes': self.num_episodes,
                        'Episodic/rews': self.rews,
                        'Episodic/rews_avg': avg_score,
                        'Episodic/ep_length': num_steps_ep
                    }
                    info.update(info_r)
                    info.pop('rew')
                    if self.num_episodes % self.log_interval == 0:
                        self.logger.scalar_summary(info, self.num_steps)
                    num_epi_offset = self.num_episodes - 1 - 10
                    if num_epi_offset % self.render_interval == 0:
                        self.logger.video_summary(tag='playback',
                                                  step=self.num_steps)
                    self.rews = 0
                    break
