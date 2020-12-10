from abc import ABC, abstractmethod
import importlib
import numpy as np
import torch.nn as nn


class AbstractModel(ABC):
    '''
    A model class should contain everything related to manipulating the
    inferred information from the network
    '''
    def __init__(self, networks):
        self.optimizer = None
        self.max_grad = None
        self.nets = list(networks)
        for net in self.nets:
            self._init_params(net)

    @staticmethod
    def _init_params(net, val=np.sqrt(2)):
        for p in net.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, val)
                p.bias.data.zero_()
            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, val)
                if p.bias is not None:
                    p.bias.data.zero_()

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad is not None:
            for net in self.nets:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    self.max_grad
                )
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
