import importlib
from types import FunctionType
from typing import Any, Iterable, T_co
from abc import abstractmethod
import itertools

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F

"""
interface that can handle most of the recent algorithms. (PG, Qlearning)
but interface itself can serve as vanilla algorithm
"""


class TorchModel(nn.Module):
    """
    input : state
    output : torch.distributions.Distribution(outputs ScalarDist if scalar output)

    this class encapsulates all neural net models + other models(e.g. decision tree) of an algorithm
    inherits nn.Module to utilize its functionalities, e.g. model.state_dict()
    child classes must init TorchModel for torch utilities
    """

    def __init__(
            self,
            observation_shape: tuple,
            action_shape: tuple,
            save_dir: str = None,
            device: str = None
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.is_save = False
        if save_dir is not None:
            self.save = True
            self.save_dir = save_dir

        available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device if device else available_device
        # self.summary_writer = SummaryWriter(log_dir=save_dir)


    @abstractmethod
    def step(self, loss):
        """
        1 single update of neural nets
        """
        pass

    @abstractmethod
    def save(self):
        """
        custom save logic
        e.g. some part of Model may use xgboost
        """
        pass

    @abstractmethod
    def load(self):
        """
        custom load logic
        """
        pass

    @staticmethod
    def init_params(net, val=np.sqrt(2)):
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, val)
                module.bias.data.zero_()
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, val)
                if module.bias is not None:
                    module.bias.data.zero_()

    @staticmethod
    def polyak_update(source, target, tau=0.95):
        for p, p_t in zip(source.parameters(), target.parameters()):
            p_t.data.copy_((1-tau) * p.data + tau * p_t.data)

    @staticmethod
    def get_optimizer_by_name(modules: Iterable, optim_name: str, **optim_kwargs) -> Optimizer:
        params = [module.parameters() for module in modules]
        mod = importlib.import_module('.'.join(optim_name.split('.')[:-1]))
        pkg = optim_name.split('.')[-1]
        optimizer = getattr(mod, pkg)(itertools.chain(*params), **optim_kwargs)

        return optimizer

    @staticmethod
    def get_loss_fn_by_name(loss_fn_name: str, **kwarg) -> FunctionType:
        src = F
        fn = loss_fn_name
        loss_fn = getattr(src, fn)

        return loss_fn


class PolicyBasedModel(TorchModel):
    """
    interface for PG models.
    must implement abstractmethods to inherit this class.
    but this class can be used as vanila policy gradient, also.
    """

    def __init__(self, observation_shape, action_shape, **kwargs):
        save_dir = kwargs.get('save_dir')
        device = kwargs.get('device')
        super().__init__(observation_shape, action_shape, save_dir=save_dir, device=device)


class ValueBasedModel(TorchModel):
    """
    interface for Q learning models.
    Q learning models always have addtional target network

    must implement abstractmethods to inherit this class.
    but this class can be used as vanila Q learning, also.
    """

    def __init__(self, observation_shape, action_shape, **kwargs):
        save_dir = kwargs.get('save_dir')
        device = kwargs.get('device')
        super().__init__(observation_shape, action_shape, save_dir, device)

        self.q_network = None
        self.target_network = None

    def update_trg(self, alpha=0.0):
        self.copy_param(self.q_network, self.target_network, alpha)

