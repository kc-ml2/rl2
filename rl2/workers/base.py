from typing import Optional
from easydict import EasyDict
from rl2.agents.base import Agent
from torch.utils.tensorboard import SummaryWriter


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
            agent: Agent,
            training=False,
            render=False,
            # render_mode: str ='human',
    ):
        self.env = env
        self.agent = agent
        self.training = training
        self.render = render
        # self.render_mode = render_mode
        self.num_episodes = 0
        self.num_steps = 0

        self.obs = env.reset()

    # def register(self, agent):
    #     self.agent.add(agent)

    def run(self):
        raise NotImplementedError

    def rollout(self):
        ac = self.agent.act(self.obs)
        obs, rew, done, info = self.env.step(ac)
        if self.training:
            info_a = self.agent.step(self.obs, ac, rew, done, obs)
            info = {**info, **info_a}
            # task_list = self.agent.dispatch()
            # if len(task_list) > 0:
            #     results = {bound_method.__name__: bound_method() for bound_method in task_list}
        else:
            # if self.render_mode:
            if self.render:
                # how to deal with render mode?
                self.env.render()
        self.num_steps += 1
        if done:  # do sth about ven env
            self.num_episodes += 1
            obs = self.env.reset()
        # Update next obs
        self.obs = obs
        info = {**info, **{'rew': rew}}
        results = None

        return done, info, results


class MaxStepWorker(RolloutWorker):
    """
    do rollout until max steps given
    """

    def __init__(self, env, agent,
                 max_steps: int, **kwargs):
        super().__init__(env, agent, **kwargs)
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
                 max_episodes: int = 10,
                 max_steps_per_ep: int = 1e4,
                 log_interval: int = 1000,
                 logger=None,
                 config=None,
                 **kwargs):
        super().__init__(env, agent, **kwargs)
        self.max_episodes = int(max_episodes)
        self.max_steps_per_ep = int(max_steps_per_ep)
        self.log_interval = log_interval
        self.num_steps_ep = 0
        self.rews = 0
        self.logger = logger
        self.config = config

    def run(self):
        for episode in range(self.max_episodes):
            while self.num_steps_ep < self.max_steps_per_ep:
                done, info, results = self.rollout()
                self.rews += info['rew']
                self.num_steps_ep += 1
                if done:
                    info_r = {
                        'Episodic/rews': self.rews
                    }
                    info.update(info_r)
                    info.pop('rew')
                    if self.num_episodes % self.log_interval == 0:
                        self.logger.scalar_summary(info, self.num_steps)
                    self.rews = 0
                    self.num_steps_ep = 0
                    break
