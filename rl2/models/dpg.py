from torch.distributions import Distribution


class DPGModel(PolicyBasedModel):
    def __init__(self, input_shape, update_target):
        pass

    def infer(self) -> Distribution:
        pass

    def step(self, loss):
        pass

    def save(self):
        pass

    def load(self):
        pass

