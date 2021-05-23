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
            self.render_mode = kwargs.get('render_mode', 'rgb_array')

        self.is_save = is_save
        if self.is_save:
            self.save_interval = save_interval

        self.num_episodes = 0
        self.num_steps = 0
        self.scores = deque(maxlen=100)
        self.ep_length = deque(maxlen=100)
        self.episode_score = np.zeros(1)
        self.ep_steps = 0 if self.n_env == 1 else np.zeros(self.n_env)

        self.obs = env.reset()
        if self.n_env > 1 and self.n_agents > 1:
            # In case there are both multi-env and multi-agents, put the
            # dimension order priority in number of agents.
            self.obs = np.array(self.obs).swapaxes(0, 1)

    def save(self, save_dir):
        for i, agent in enumerate(self.agents):
            _save_dir = os.path.join(save_dir,
                                    f'ckpt/agent{i}/{self.num_steps//1000}k')
            Path(_save_dir).mkdir(parents=True, exist_ok=True)
            agent.model.save(_save_dir)

    def rollout(self):
        # Number of agents should always match with number of dimensions
        # returned and possibly the dones.
        acs = [agent.act(obs) for agent, obs in zip(self.agents, self.obs)]
        if self.n_env > 1 and self.n_agents > 1:
            acs = np.array(acs).swapaxes(0, 1)

        obss, rews, dones, info = self.env.step(acs)
        self.episode_score = self.episode_score + np.array(rews)
        if self.n_env > 1 and self.n_agents > 1:
            # Swap axes so that the first dimension is agents
            obss = np.array(obss).swapaxes(0, 1)
            acs = np.array(acs).swapaxes(0, 1)
            rews = np.asarray(rews).swapaxes(0, 1)
            dones = np.asarray(dones).swapaxes(0, 1)

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

        self.num_steps += self.n_env
        dones = np.asarray(dones)
        self.ep_steps = self.ep_steps + 1
        if self.n_env == 1:
            if self.n_agents > 1:
                dones = all(dones)
            if dones:
                self.num_episodes += 1
                obss = self.env.reset()
                self.scores.append(self.episode_score.mean())
                self.episode_score = 0.
                self.ep_length.append(self.ep_steps)
                self.ep_steps = 0
        else:
            if self.n_agents > 1:
                dones = dones.all(axis=0)
            # Vector env + centralized
            self.num_episodes += sum(dones)
            for d_i in np.where(dones)[0]:
                self.scores.append(self.episode_score[d_i].mean())
                self.episode_score[d_i] = np.zeros_like(
                    self.episode_score[d_i])
                self.ep_length.append(self.ep_steps[d_i])
                self.ep_steps[d_i] = 0

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
            self.render_mode = kwargs.get('render_mode', 'rgb_array')

        self.is_save = is_save
        if self.is_save:
            self.save_interval = save_interval

        self.num_episodes = 0
        self.num_steps = 0
        self.scores = deque(maxlen=100)
        self.ep_length = deque(maxlen=100)
        self.episode_score = np.zeros(1)
        self.ep_steps = 0 if self.n_env == 0 else np.zeros(self.n_env)

        self.obs = env.reset()
        self.obs = np.asarray(self.obs)
        if self.n_env > 1:
            # Squash first two dimensions to make them like a batch
            self.obs = self.obs.reshape((-1, *self.obs.shape[2:]))

    def save(self, save_dir):
        save_dir = os.path.join(save_dir, f'ckpt/{int(self.num_steps/1000)}k')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.agent.model.save(save_dir)

    def rollout(self):
        # Number of agents should always match with number of dimensions
        # returned and possibly the dones.
        acs = self.agent.act(self.obs).reshape(
            self.n_env, self.n_agents, -1).squeeze()

        obss, rews, dones, info = self.env.step(acs)
        self.episode_score = self.episode_score + np.array(rews)
        obss = np.asarray(obss)
        if self.n_env > 1 and self.n_agents > 1:
            # Swap axes so that the first dimension is agents
            obss = obss.reshape(-1, *obss.shape[2:])
            acs = acs.flatten()
            rews = rews.flatten()
            ep_dones = dones
            dones = dones.flatten()

        infos = {}
        if isinstance(info, list) or isinstance(info, tuple):
            infos = info[0]
        else:
            infos = info

        if self.training:
            self.agent.step(self.obs, acs, rews, dones, obss)

        self.num_steps += self.n_env
        self.ep_steps = self.ep_steps + 1
        if self.n_env == 1:
            if self.n_agents > 1:
                dones = all(dones)
            if dones:
                self.num_episodes += 1
                obss = self.env.reset()
                obss = np.asarray(obss)
                self.scores.append(self.episode_score.mean())
                self.episode_score = 0.
                self.ep_length.append(self.ep_steps)
                self.ep_steps = 0
        else:
            if self.n_agents > 1:
                # dones = dones.reshape(self.n_agents, self.n_env)
                dones = ep_dones.all(axis=1)
            # Vector env + centralized
            self.num_episodes += sum(dones)
            for d_i in np.where(dones)[0]:
                self.scores.append(self.episode_score[d_i].mean())
                self.episode_score[d_i] = np.zeros_like(
                    self.episode_score[d_i])
                self.ep_length.append(self.ep_steps[d_i])
                self.ep_steps[d_i] = 0

        # Update next obs
        self.obs = obss
        results = None

        return dones, infos, results


def dynamic_class(cls1, cls2, *args, **kwargs):
    class CombinedClass(cls1, cls2):
        def __init__(self, *args, **kwargs):
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
