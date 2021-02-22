from rl2.agents.base import Agent
from rl2.models.torch.base import ActorCriticModel


class A2CModel(ActorCriticModel):
    """
    predefined model
    (same one as original paper)
    """
    pass


class A2CAgent(Agent):
    def __init__(self):
        pass

    def train(self, batch):
        pass

    def act(self, obs):
        pass

    def collect(self):
        pass
