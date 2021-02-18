from rl2.agents.base import Agent


class RolloutWorker:
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

        self.obs = env.reset()

    # def register(self, agent):
    #     self.agent.add(agent)

    def run(self):
        raise NotImplementedError

    def rollout(self):
        ac = self.agent.act(self.obs)
        obs, rew, done, info = self.env.step(ac)
        if self.training:
            self.agent.collect(obs, ac, rew, done, self.obs)
            self.agent.step()
            # task_list = self.agent.dispatch()
            # if len(task_list) > 0:
            #     results = {bound_method.__name__: bound_method() for bound_method in task_list}
        else:
            # if self.render_mode:
            if self.render:
                # how to deal with render mode?
                self.env.render()

        self.obs = obs

        return done, info, results


class MaxStepWorker(RolloutWorker):
    def __init__(self, env, agent, max_steps, **kwargs):
        super().__init__(env, agent, **kwargs)
        self.max_steps = max_steps

    def run(self):
        for step in range(self.max_steps):
            done, info, results = self.rollout()

            if done:  # do sth about ven env
                self.num_episodes += 1

            # TODO: when done do sth like logging from results


class EpisodicWorker(RolloutWorker):
    def __init__(self, env, agent, num_episodes=1, **kwargs):
        super().__init__(env, agent, **kwargs)
        self.num_episodes = num_episodes

    def run(self):
        for episode in range(self.num_episodes):
            done = False
            while not done:
                done, info = self.rollout()

            # TODO: when done do sth like logging
