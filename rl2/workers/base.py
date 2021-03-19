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
        self.render_interval = render_interval
        self.is_save = is_save
        if self.is_save:
            self.save_interval = save_interval

        self.render_mode = kwargs.get('render_mode')
        self.num_episodes = 0
        self.num_steps = 0
        self.scores = deque(maxlen=100)
        self.episode_score = 0

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
                    for env_i, info_i in enumerate(info):
                        info['env_{}'.format(env_i)] = info_i
            else:
                info = {**info_a}
            # task_list = self.agent.dispatch()
            # if len(task_list) > 0:
            #     results = {bound_method.__name__: bound_method() for bound_method in task_list}
            if self.render:
                # how to deal with render mode?
                if self.render_mode == 'rgb_array' and self.num_episodes % self.render_interval < 10:
                    self._rendering = True
                    self._num_rendering = 0
                    rgb_array = self.env.render('rgb_array')
                    self.logger.store_rgb(rgb_array)
                elif self.render_mode == 'ascii':
                    self.env.render(self.render_mode)
                elif self.render_mode == 'human':
                    self.env.render()
        # else:
            # if self.render_mode:
        self.num_steps += self.n_env
        self.episode_score = self.episode_score + np.array(rew)
        if isinstance(done, Iterable):
            if any(done):
                self.num_episodes += sum(done)
                for d_i, d in enumerate(done):
                    if d:
                        self.scores.append(self.episode_score[d_i])
                        self.episode_score[d_i] = 0.0
        else:
            if done:  # do sth about ven env
                self.num_episodes += 1
                obs = self.env.reset()
                self.scores.append(self.episode_score)
                self.episode_score = 0.
        # Update next obs
        self.obs = obs
        info = {**info, **{'rew': rew}}
        results = None

        # Save model
        if self.is_save and self.num_steps % self.save_interval == 0:
            if hasattr(self, 'logger'):
                save_dir = getattr(self.logger, 'log_dir')
            self.save(save_dir)

        return done, info, results


class MaxStepWorker(RolloutWorker):
    """
    do rollout until max steps given
    """

    def __init__(self, env, n_env, agent,
                 max_steps: int, **kwargs):
        super().__init__(env, n_env, agent, **kwargs)
        self.max_steps = int(max_steps)
        self.log_step = 5000

    def run(self):
        episode_score = 0.0
        for step in range(self.max_steps // self.n_env + 1):
            done, info, results = self.rollout()

            if step * self.n_env % self.log_step < self.n_env and self.num_episodes > 0:
                avg_score = sum(list(self.scores)) / len(list(self.scores))
                print(step * self.n_env, self.num_episodes, avg_score)


class EpisodicWorker(RolloutWorker):
    """
    do rollout until max episodes given
    might be useful at inference time or when training episodically
    """

    def __init__(self, env, n_env, agent,
                 max_episodes: int = 10,
                 max_steps_per_ep: int = 1e4,
                 log_interval: int = 1000,
                 logger=None,
                 **kwargs):
        super().__init__(env, n_env, agent, **kwargs)
        self.max_episodes = int(max_episodes)
        self.max_steps_per_ep = int(max_steps_per_ep)
        self.log_interval = log_interval
        self.num_steps_ep = 0
        self.rews = 0
        self.scores = deque(maxlen=100)
        self.logger = logger

    def run(self):
        for episode in range(self.max_episodes):
            while self.num_steps_ep < self.max_steps_per_ep:
                done, info, results = self.rollout()
                self.rews += info['rew']
                self.num_steps_ep += 1
                if done:
                    self.scores.append(self.rews)
                    avg_score = sum(list(self.scores)) / len(list(self.scores))
                    info_r = {
                        'Counts/num_steps': self.num_steps,
                        'Counts/num_episodes': self.num_episodes,
                        'Episodic/rews': self.rews,
                        'Episodic/rews_avg': avg_score,
                        'Episodic/ep_length': self.num_steps_ep
                    }
                    info.update(info_r)
                    info.pop('rew')
                    if self.num_episodes % self.log_interval == 0:
                        self.logger.scalar_summary(info, self.num_steps)
                    if ((self.num_episodes-1) - 10) % self.render_interval == 0:
                        self.logger.video_summary(tag='playback',
                                                  step=self.num_steps)
                    self.rews = 0
                    self.num_steps_ep = 0
                    break
