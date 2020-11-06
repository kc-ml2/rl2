from abc import ABC, abstractmethod
import importlib


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

    def set_optimizer(self, modules, optimizer, optim_args):
        params = []
        for module in modules:
            params = params + list(module.parameters())
        mod = importlib.import_module('.'.join(optimizer.split('.')[:-1]))
        pkg = optimizer.split('.')[-1]
        self.optimizer = getattr(mod, pkg)(
            params,
            **optim_args
        )
