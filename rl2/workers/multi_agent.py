import os
from pathlib import Path
from typing import List
from rl2.agents.base import Agent
from rl2.workers.base import MaxStepWorker, EpisodicWorker
import numpy as np
from collections import deque


class MultiAgentRolloutWorker:
    def __init__(
        self,
        env,
        n_env,
        agents: List[Agent],
        training=False,
        render=False,
        render_interval=1000,
        is_save=False,
        save_interval=int(1e6),
        **kwargs
    ):
        self.env = env
        self.n_env = n_env
        self.agents = agents
        self.n_agents = len(agents)
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
        self.scores = deque(maxlen=100)
        self.ep_length = deque(maxlen=100)
        self.episode_score = 0
        self.ep_steps = np.zeros(self.n_env)

        self.obs = env.reset()
        if self.n_env > 1 and self.n_agents > 1:
            # In case there are both multi-env and multi-agents, put the
            # dimension order priority in number of agents.
            self.obs = np.array(self.obs).swapaxes(0, 1)

    def save(self, save_dir):
        for i, agent in enumerate(self.agents):
            save_dir = os.path.join(save_dir,
                                    f'ckpt/agent{i}/{self.num_steps//1000}k')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            agent.model.save(save_dir)

    def rollout(self):
        # Number of agents should always match with number of dimensions
        # returned and possibly the dones.
        acs = [agent.act(obs) for agent, obs in zip(self.agents, self.obs)]
        if self.n_env > 1 and self.n_agents > 1:
            acs = np.array(acs).swapaxes(0, 1)

        obss, rews, dones, info = self.env.step(acs)
        if self.n_env > 1 and self.n_agents > 1:
            ep_rew = np.asarray(rews).sum(1)
            # Swap axes so that the first dimension is agents
            obss = np.array(obss).swapaxes(0, 1)
            acs = np.array(acs).swapaxes(0, 1)
            rews = np.asarray(rews).swapaxes(0, 1)
            dones = np.asarray(dones).swapaxes(0, 1)
        elif self.n_agents > 1:
            ep_rew = np.asarray(rews).sum()
        else:
            ep_rew = np.asarray(rews)

        infos = {}
        if isinstance(info, list) or isinstance(info, tuple):
            infos = info[0]
        else:
            infos = info

        if self.training:
            for agent, obs, ac, rew, done, obs_ in zip(
                    self.agents, self.obs, acs, rews, dones, obss):
                info_a = agent.step(obs, ac, rew, done, obs_)
                infos.update(info_a)

        if self.render:
            infos['image'] = self.env.render(self.render_mode)

        self.num_steps += self.n_env
        self.episode_score = self.episode_score + np.array(ep_rew)
        dones = np.asarray(dones)
        self.ep_steps = self.ep_steps + 1
        if self.n_env == 1:
            if self.n_agents > 1:
                dones = all(dones)
            if dones:
                self.num_episodes += 1
                obss = self.env.reset()
                self.scores.append(self.episode_score)
                self.episode_score = 0.
        else:
            if self.n_agents > 1:
                dones = dones.all(axis=0)
            # Vector env + centralized
            self.num_episodes += sum(dones)
            for d_i in np.where(dones)[0]:
                self.scores.append(self.episode_score[d_i])
                self.episode_score[d_i] = np.zeros_like(
                    self.episode_score[d_i])
                self.ep_length.append(self.ep_steps[d_i])
                self.ep_steps[d_i] = np.zeros_like(self.ep_steps[d_i])

        # Update next obs
        self.obs = obss
        results = None

        return dones, infos, results


class SelfRolloutWorker:
    def __init__(
        self,
        env,
        n_env,
        agent,
        n_agents=0,
        training=False,
        render=False,
        render_interval=1000,
        is_save=False,
        save_interval=int(1e6),
        **kwargs
    ):
        self.env = env
        self.n_env = n_env
        self.agent = agent
        assert n_agents > 0, "Must provide n_agents"
        self.n_agents = n_agents
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
        self.scores = deque(maxlen=100)
        self.ep_length = deque(maxlen=100)
        self.episode_score = 0
        self.ep_steps = np.zeros(self.n_env)

        self.obs = env.reset()
        if self.n_env > 1:
            # Squash first two dimensions to make them like a batch
            self.obs = np.array(self.obs)
            self.obs = self.obs.rehape((-1, *self.obs.shape[2:]))

    def save(self, save_dir):
        save_dir = os.path.join(save_dir, f'ckpt/{int(self.num_steps/1000)}k')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.agent.model.save(save_dir)

    def rollout(self):
        # Number of agents should always match with number of dimensions
        # returned and possibly the dones.
        acs = self.agent(self.obs).reshape(
            self.n_env, self.n_agents, -1).squeeze()

        obss, rews, dones, info = self.env.step(acs)
        if self.n_env > 1 and self.n_agents > 1:
            ep_rew = np.asarray(rews).mean(1)
            # Swap axes so that the first dimension is agents
            obss = np.asarray(obss).reshape(-1, obss.shape[2:])
            acs = acs.flatten()
            rews = rews.flatten()
            dones = dones.flatten()
        elif self.n_agents > 1:
            ep_rew = np.asarray(rews).mean()
        else:
            ep_rew = np.asarray(rews)

        if self.training:
            self.agent.step(self.obs, acs, rews, dones, obss)

        if self.render:
            info['image'] = self.env.render(self.render_mode)

        self.num_steps += self.n_env
        self.episode_score = self.episode_score + np.array(ep_rew)
        self.ep_steps = self.ep_steps + 1
        if self.n_env == 1:
            if self.n_agents > 1:
                dones = all(dones)
            if dones:
                self.num_episodes += 1
                obss = self.env.reset()
                self.scores.append(self.episode_score)
                self.episode_score = 0.
        else:
            if self.n_agents > 1:
                dones = dones.all(axis=0)
            # Vector env + centralized
            self.num_episodes += sum(dones)
            for d_i in np.where(dones)[0]:
                self.scores.append(self.episode_score[d_i])
                self.episode_score[d_i] = np.zeros_like(
                    self.episode_score[d_i])
                self.ep_length.append(self.ep_steps[d_i])
                self.ep_steps[d_i] = np.zeros_like(self.ep_steps[d_i])

        # Update next obs
        self.obs = obss
        results = None

        return dones, info, results


# class MaxStepWorker(RolloutWorker):
#     """
#     do rollout until max steps given
#     """
#
#     def __init__(self, env, agent,
#                  max_steps=None, **kwargs):
#         super().__init__(env, agent, **kwargs)
#         assert max_steps is not None, 'must provide max_steps'
#         self.max_steps = int(max_steps)
#
#     def run(self):
#         for step in range(self.max_steps):
#             done, info, results = self.rollout()
#
#             # TODO: when done do sth like logging from results


# class EpisodicWorker(RolloutWorker):
#     """
#     do rollout until max episodes given
#     might be useful at inference time or when training episodically
#     """
#     def __init__(self, env, agent,
#                  max_steps: int = None,
#                  max_episodes: int = 10,
#                  max_steps_per_ep: int = 1e4,
#                  **kwargs):
#         super().__init__(env, agent, **kwargs)
#         self.max_steps = int(max_steps)
#         self.max_episodes = int(max_episodes)
#         self.max_steps_per_ep = int(math.inf) if max_steps is None else int(
#             max_steps_per_ep)
#         self.rews = 0
#         self.rews_ep = []

#     def run(self):
#         for episode in range(self.max_episodes):
#             for step in range(self.max_steps_per_ep):
#                 done, info, results = self.rollout()
#                 self.rews += info
#                 if done or step == (self.max_steps-1):
#                     self.rews_ep.append(self.rews)
#                     print(
#                         f"num_ep: {self.num_episodes}, "
#                         "episodic_reward: {self.rews}")
#                     self.rews = 0


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
        self.rews = np.zeros(len(self.agents))
        self.scores = [deque(maxlen=100) for _ in range(len(self.agents))]
        self.logger = logger
        self.n_env = n_env
        self.obss = self.obs
        if self.n_env > 1:
            self.obss = np.swapaxes(np.asarray(self.obs), 0, 1)
        self.infos = [{} for _ in range(len(self.agents))]

    def rollout(self):
        infos = [{} for _ in range(len(self.agents))]
        acs = [agent.act(obs) for agent, obs in zip(self.agents, self.obss)]

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
                        self.infos[i].update(info_a)
        if self.render:
            if (self.render_mode == 'rgb_array' and
                    self.num_episodes % self.render_interval < 10):
                rgb_array = self.env.render('rgb_array')
                self.logger.store_rgb(rgb_array)
            elif self.render_mode == 'ascii':
                self.env.render(self.render_mode)
            elif self.render_mode == 'human':
                self.env.render()

        self.num_steps += self.n_env
        if all(dones):
            self.num_episodes += 1
            if self.n_env > 1:
                dones = dones[:, 0]
            else:
                obss = self.env.reset()
        # Update next obs
        self.obss = obss
        results = None

        # Save model
        if self.is_save and self.num_steps % self.save_interval < self.n_env:
            if hasattr(self, 'logger'):
                save_dir = getattr(self.logger, 'log_dir')
            self.save(save_dir)

        return dones, infos, results

    def run(self):
        for episode in range(self.max_episodes):
            for num_steps_ep in range(self.max_steps_per_ep):
                dones, infos, results = self.rollout()
                for i in np.where(dones)[0]:
                    # Agent specific values to log
                    self.scores[i].append(self.rews[i])
                    avg_score = np.mean(list(self.scores[i]))
                    info_r = {
                        'Counts/agent_steps': self.agents[i].curr_step,
                        'Episodic/rews': self.rews[i],
                        'Episodic/rews_avg': avg_score,
                        'Episodic/ep_length': num_steps_ep
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
                        if all([agent.curr_step > agent.update_after
                               for agent in self.agents]):
                            self.logger.scalar_summary(summary, self.num_steps)
                    if self.render:
                        num_epi_offset = self.num_episodes - 1 - 10
                        if num_epi_offset % self.render_interval == 0:
                            self.logger.video_summary(tag='playback',
                                                      step=self.num_steps)

                    # Reset variables
                    self.rews = np.zeros(len(self.agents))
                    break

    def save(self, save_dir):
        prefix = save_dir
        for i, agent in enumerate(self.agents):
            save_dir = os.path.join(
                prefix, f'agent_{i}/ckpt/{int(self.num_steps/1000)}k')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            agent.model.save(save_dir)


class CentralizedEpisodicWorker(EpisodicWorker):
    def __init__(self, env, n_env, agent,
                 max_episodes: int,
                 max_steps_per_ep: int,
                 log_interval: int,
                 logger,
                 **kwargs):
        super().__init__(env, n_env, agent,
                         max_episodes=max_episodes,
                         max_steps_per_ep=max_steps_per_ep,
                         log_interval=log_interval,
                         logger=logger,
                         render_interval=50,
                         render_mode='rgb_array',
                         **kwargs)
        self.info = {}

        if self.n_env > 1:
            self.obs = self.obs.reshape(-1, *
                                        self.agent.model.observation_shape)

    def rollout(self):
        acs = []
        info = {}
        acs = self.agent.act(self.obs)

        if self.n_env > 1:
            acs = acs.reshape(-1, 4)
        obs, rews, dones, info_e = self.env.step(acs)
        if self.n_env > 1:
            obs = obs.reshape(-1, *self.agent.model.observation_shape)
            rews = rews.reshape(-1,)
            dones = dones.reshape(-1,)
            acs = acs.reshape(-1,)
            self.rews += rews.mean(-1)
        else:
            self.rews += rews
        if self.training:
            if self.n_env > 1:
                info_a = self.agent.step(self.obs, acs, rews, dones, obs)
                self.info.update(info_a)
        if self.render:
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
            # FIXME: num_snakes?
            dones = dones.reshape(self.n_env, 4)
            dones = dones[0, :]
            if all(dones):
                self.num_episodes += 1
        else:
            if all(dones):  # do sth about ven env
                self.num_episodes += 1
                obs = self.env.reset()
        # Update next obs
        self.obs = obs
        results = None

        # Save model
        if self.is_save and self.num_steps % self.save_interval == self.n_env:
            if hasattr(self, 'logger'):
                save_dir = getattr(self.logger, 'log_dir')
            self.save(save_dir)

        return dones, info, results

    def run(self):
        for episode in range(self.max_episodes):
            for num_steps_ep in range(self.max_steps_per_ep):
                dones, infos, results = self.rollout()
                # Agent specific values to log
                self.scores.append(self.rews)
                avg_score = np.mean(list(self.scores))
                info_r = {
                    'Counts/agent_steps': self.agent.curr_step,
                    'Episodic/rews': self.rews,
                    'Episodic/rews_avg': avg_score,
                    'Episodic/ep_length': num_steps_ep
                }
                self.info.update(info_r)
                if all(dones):
                    if self.num_episodes % self.log_interval == 0:
                        summary = {}
                        # Global value to log
                        counts = {'Counts/num_steps': self.num_steps,
                                  'Counts/num_episodes': self.num_episodes}
                        summary.update(counts)
                        summary.update(self.info)
                        self.logger.scalar_summary(summary, self.num_steps)
                    if self.render:
                        num_epi_offset = self.num_episodes - 1 - 10
                        if num_epi_offset % self.render_interval == 0:
                            self.logger.video_summary(tag='playback',
                                                      step=self.num_steps)

                    # Reset variables
                    self.rews = 0
                    break


def dynamic_class(cls1, cls2, *args, **kwargs):
    class CombinedClass(cls1, cls2):
        def __init__(self, *args, **kwargs):
            # super(combinedclass, self).__init__(*args, **kwargs)
            cls2.__init__(self, *args, **kwargs)
            cls1.__init__(self, *args, **kwargs)

    return CombinedClass(*args, **kwargs)


def MAMaxStepWorker(env, n_env, agent, **kwargs):
    return dynamic_class(MultiAgentRolloutWorker, MaxStepWorker,
                         env, n_env, agent, **kwargs)


def SelfMaxStepWorker(env, n_env, agent, **kwargs):
    return dynamic_class(SelfRolloutWorker, MaxStepWorker,
                         env, n_env, agent, **kwargs)


def MAEpisodicWorker(env, n_env, agent, **kwargs):
    return dynamic_class(MultiAgentRolloutWorker, EpisodicWorker,
                         env, n_env, agent, **kwargs)


def SelfEpisodicWorker(env, n_env, agent, **kwargs):
    return dynamic_class(SelfRolloutWorker, EpisodicWorker,
                         env, n_env, agent, **kwargs)
