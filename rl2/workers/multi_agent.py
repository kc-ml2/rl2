class MultiAgentRolloutWorker:
    def __init__(self, num_agents):
        self.num_agents = num_agents


class MaxStepWorker:
    def __init__(
            self, env, pretrained_agents, training_agents,
            max_steps, **kwargs
    ):
        self.pretrained_agents = pretrained_agents
        self.training_agents = training_agents

        # order
        self.agents = self.pretrained_agents + self.training_agents

        self.max_steps = max_steps

        super().__init__(env)

    def run(self):
        obs = self.env.reset()
        # for episode in max_episodes
        for step in range(self.max_steps):
            acs = [agent.act(obs) for agent in self.agents]
            obs, rew, done, info = self.env.step(acs)

            for agent in self.training_agents:
                agent.collect(obs, rew, done, info)

                if at_some_point:
                    self.agent.train()


class SelfPlayWorker:
    def __init__(self, env, agent):
        pass
