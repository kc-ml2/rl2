import math
from rl2.workers.base import RolloutWorker


class MultiAgentRolloutWorker:
    def __init__(
        self,
        env,
        agents,
        training=False,
        render=False,
        **kwargs
    ):
        self.env = env
        self.agents = agents
        self.training = training
        self.render = render
        # self.render_mode = render_mode
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
                agent.step(obs, ac, rew, done, obs)
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
