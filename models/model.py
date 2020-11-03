from abc import ABC, abstractmethod


class AbstractModel(ABC):
    '''
    A model class should contain everything related to manipulating the
    inferred information from the network
    '''
    def __init__(self):
        self.optimizer = None

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @abstractmethod
    def infer(self, x):
        pass
