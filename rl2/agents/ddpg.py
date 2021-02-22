from torch.distributions import Distribution

from rl2.agents.base import Agent
from rl2.models.torch.dpg import DPGModel


class DDPGModel(DPGModel):
    """
    predefined model
    (same one as original paper)
    """

    def __init__(self, input_shape, **kwargs):
        pass

    def infer(self) -> Distribution:
        pass

    def step(self, loss):
        pass

    def save(self):
        pass

    def load(self):
        pass


class DDPGAgent(Agent):
    def __init__(self, model: DDPGModel, **kwargs):
        config = kwargs['config']
        # prioritized experience replay
        # self.per = config.pop('per', False)
        # self.buffer = PrioritizedReplayBuffer() if self.per else ReplayBuffer()

        super().__init__(**kwargs)

    def act(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def collect(self):
        pass
