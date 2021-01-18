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

    @abstractmethod
    def infer(self, x):
        pass

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

    @staticmethod
    def _copy_param(source, target, alpha=0.0):
        for p, p_t in zip(source.parameters(), target.parameters()):
            p_t.data.copy_(alpha * p_t.data + (1 - alpha) * p.data)

    @staticmethod
    def set_optimizer(modules, optimizer, optim_args):
        params = []
        for module in modules:
            params = params + list(module.parameters())
        mod = importlib.import_module('.'.join(optimizer.split('.')[:-1]))
        pkg = optimizer.split('.')[-1]
        optimizer = getattr(mod, pkg)(
            params,
            **optim_args
        )
        return optimizer

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
