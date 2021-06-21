from torch.distributions import Distribution

from rl2.models.base import PolicyBasedModel


class DPGModel(PolicyBasedModel):
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

