import math
from typing import List
from rl2.agents.base import Agent
from rl2.workers.base import RolloutWorker
import numpy as np
from collections import deque, ChainMap


class MultiAgentRolloutWorker:
    def __init__(
        self,
        env,
        agents: List[Agent],
        training=False,
        render=False,
        render_interval=1000,
        is_save=False,
        save_interval=int(1e6),
        **kwargs
    ):
        self.env = env
        self.agents = agents
        self.training = training

        self.render = render
        if self.render:
            self.render_interval = render_interval
            self.render_mode = kwargs.get('render_mode')

        self.is_save = is_save
        if self.is_save:
            self.save_interval = save_interval

        self.num_episodes = 0
        self.num_steps = 0

        self.obs = env.reset()

    def run(self):
        raise NotImplementedError

    def rollout(self):
        acs = []
        for agent in self.agents:
            ac = agent.act(self.obs)
            acs.append(ac)

        obss, rews, dones, info = self.env.step(acs)
        if self.training:
            for agent, obs, ac, rew, done, obs_ in (
                    self.agents, self.obs, acs, rews, dones, obss):
                agent.step(obs, ac, rew, done, obs_)
        else:
            if self.render:
                self.env.render()
        self.num_steps += 1
        if all(dones):  # do sth about ven env
            self.num_episodes += 1
            obss = self.env.reset()
        # Update next obs
        self.obs = obss
        info = rews
        results = None

        return dones, info, results


class SelfRolloutWorker(RolloutWorker):
    def __init__(
        self,
        env,
        agents,
        training=False,
        render=False,
        **kwargs
    ):
        super().__init__(env, agents, training=training, render=render)

    def rollout(self):
        ac = self.agent.act(self.obs)
        obs, rew, done, info = self.env.step(ac)
        if self.training:
            self.agent.step(self.obs, ac, rew, done, obs)
        else:
            if self.render:
                # how to deal with render mode?
                self.env.render()
        self.num_steps += 1
        if all(done):  # do sth about ven env
            self.num_episodes += 1
            obs = self.env.reset()
        # Update next obs
        self.obs = obs
        info = sum(rew)
        results = None

        return done, info, results


class MaxStepWorker(RolloutWorker):
    """
    do rollout until max steps given
    """

    def __init__(self, env, agent,
                 max_steps=None, **kwargs):
        super().__init__(env, agent, **kwargs)
        assert max_steps is not None, 'must provide max_steps'
        self.max_steps = int(max_steps)

    def run(self):
        for step in range(self.max_steps):
            done, info, results = self.rollout()

            # TODO: when done do sth like logging from results


class EpisodicWorker(RolloutWorker):
    """
    do rollout until max episodes given
    might be useful at inference time or when training episodically
    """

    def __init__(self, env, agent,
                 max_steps: int = None,
                 max_episodes: int = 10,
                 max_steps_per_ep: int = 1e4,
                 **kwargs):
        super().__init__(env, agent, **kwargs)
        self.max_steps = int(max_steps)
        self.max_episodes = int(max_episodes)
        self.max_steps_per_ep = int(math.inf) if max_steps is None else int(
            max_steps_per_ep)
        self.rews = 0
        self.rews_ep = []

    def run(self):
        for episode in range(self.max_episodes):
            for step in range(self.max_steps_per_ep):
                done, info, results = self.rollout()
                self.rews += info
                if done or step == (self.max_steps-1):
                    self.rews_ep.append(self.rews)
                    print(
                        f"num_ep: {self.num_episodes}, "
                        "episodic_reward: {self.rews}")
                    self.rews = 0

        # TODO: when done do sth like logging


class IndividualEpisodicWorker(MultiAgentRolloutWorker):
    """
    Rollout worker for individual agents
    i.e. Fully decentralized agents
    """

    def __init__(self,
                 env,
                 agents,
                 max_episodes: int = 10,
                 max_steps_per_ep: int = int(1e4),
                 log_interval: int = 1000,
                 logger=None,
                 n_env=1,
                 **kwargs):
        super().__init__(env, agents, **kwargs)
        self.max_episodes = int(max_episodes)
        self.max_steps_per_ep = int(max_steps_per_ep)
        self.log_interval = int(log_interval)
        self.render_mode = kwargs.setdefault('render_mode', 'rgb_array')
        self.num_steps_ep = 0
        self.rews = np.zeros(len(self.agents))
        self.scores = [deque(maxlen=100) for _ in range(len(self.agents))]
        self.logger = logger
        self.n_env = n_env
        self.obss = self.obs
        if self.n_env > 1:
            self.obss = np.swapaxes(np.asarray(self.obs), 0, 1)
        self.infos = [{} for _ in range(len(self.agents))]

    def rollout(self):
        acs = []
        infos = [{} for _ in range(len(self.agents))]
        # infos = self.infos
        for agent, obs in zip(self.agents, self.obss):
            ac = agent.act(obs)
            acs.append(ac)

        if self.n_env > 1:
            acs = np.array(acs).T
        obss, rews, dones, info_e = self.env.step(acs)
        if self.n_env > 1:
            obss = np.swapaxes(np.asarray(obss), 0, 1)
            rews = np.swapaxes(np.asarray(rews), 0, 1)
            dones = np.swapaxes(np.asarray(dones), 0, 1)
            acs = acs.T
            self.rews += rews.mean(-1)
        else:
            self.rews += rews
        if self.training:
            for i, (agent, obs, ac, rew, done, obs_) in enumerate(zip(
                    self.agents, self.obss, acs, rews, dones, obss)):
                if self.n_env > 1:
                    info_a = agent.step(obs, ac, rew, done, obs_)
                    self.infos[i].update(info_a)
                else:
                    if not done:
                        info_a = agent.step(obs, ac, rew, done, obs_)
                        # FIXME: Handle Vector Env
                        # if info_e:
                        #     if isinstance(info_e, dict):
                        #         infos[i] = {**infos[i], **info_e}
                        #     elif isinstance(info_e, 'list') or isinstance(info_e, 'tuple'):
                        #         infos[i] = {**info_a}
                        #         for env_i, info_i in enumerate(info_e):
                        #             infos[i]['env_{}'.format(env_i)] = info_i
                        #     infos[i].update(info_a)
                        # else:
                        # for key in list(info_a.keys()):
                        #     info_a['agent_{}/'.format(i)+key] = info_a.pop(key)
                        self.infos[i].update(info_a)
        if self.render:
            # how to deal with render mode?
            # FIXME: deal with render interval for MaxStepWorker
            if (self.render_mode == 'rgb_array' and
                self.num_episodes % self.render_interval < 10):
                rgb_array = self.env.render('rgb_array')
                # Push rgb_array to logger's buffer
                self.logger.store_rgb(rgb_array)
            elif self.render_mode == 'ascii':
                self.env.render(self.render_mode)
            elif self.render_mode == 'human':
                self.env.render()

        self.num_steps += self.n_env
        if self.n_env > 1:
            dones = dones[:, 0]
            if all(dones):
                self.num_episodes += 1
        else:
            if all(dones):  # do sth about ven env
                self.num_episodes += 1
                obss = self.env.reset()
        # Update next obs
        self.obss = obss
        results = None

        return dones, infos, results

    def run(self):
        for episode in range(self.max_episodes):
            while self.num_steps_ep < self.max_steps_per_ep:
                dones, infos, results = self.rollout()
                self.num_steps_ep += 1
                for i in np.where(dones)[0]:
                    # Agent specific values to log
                    self.scores[i].append(self.rews[i])
                    avg_score = np.mean(list(self.scores[i]))
                    info_r = {
                        'Counts/agent_steps': self.agents[i].curr_step,
                        'Episodic/rews': self.rews[i],
                        'Episodic/rews_avg': avg_score,
                        'Episodic/ep_length': self.num_steps_ep
                    }
                    self.infos[i].update(info_r)

                if all(dones):
                    if self.num_episodes % self.log_interval == 0:
                        summary = {}
                        for i, info in enumerate(self.infos):
                            for k, v in info.items():
                                if k not in summary.keys():
                                    summary.update(
                                        {k: {'agent_{}'.format(i): v}})
                                else:
                                    summary[k].update(
                                        {'agent_{}'.format(i): v})
                        # Global value to log
                        counts = {'Counts/num_steps': self.num_steps,
                                  'Counts/num_episodes': self.num_episodes}
                        summary.update(counts)
                        if all([agent.curr_step > agent.update_after for agent in self.agents]):
                            self.logger.scalar_summary(summary, self.num_steps)
                    if self.render:
                        if ((self.num_episodes-1) - 10) % self.render_interval == 0:
                            self.logger.video_summary(tag='playback',
                                                    step=self.num_steps)

                    # Reset variables
                    self.rews = np.zeros(len(self.agents))
                    self.num_steps_ep = 0
                    # self.infos = [{} for _ in range(len(self.agents))]
                    break


def dynamic_class(cls1, cls2, *args, **kwargs):
    class CombinedClass(cls1, cls2):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    return CombinedClass(*args, **kwargs)


def MAMaxStepWorker(env, agent, **kwargs):
    return dynamic_class(MaxStepWorker, MultiAgentRolloutWorker,
                         env, agent, **kwargs)


def SelfMaxStepWorker(env, agent, **kwargs):
    return dynamic_class(MaxStepWorker, SelfRolloutWorker,
                         env, agent, **kwargs)
