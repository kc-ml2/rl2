from typing import Union, List
import numpy as np

from rl2.models.tf.base import TFModel
from rl2.models.torch.base import TorchModel


class Agent:
    """
    interface.
    agent has buffer
    even pg algorhtms use buffer(not the true experience replay buffer but it triggers train when buffer is full)
    agent counts interactions and do what it needs to do when it needs to be done
    """

    def __init__(
            self,
            # model must be instantiated and initialized before passing as argument
            model: Union[TorchModel, TFModel],
            train_interval,
            num_epochs,
            # device,
            buffer_cls,
            buffer_kwargs,
    ):
        self.curr_step = 0
        self.train_interval = train_interval

        # """
        self.model = model

        self.num_epochs = num_epochs
        # self.device = device

        self.buffer = buffer_cls(**buffer_kwargs)
        self._hook = None

    @property
    def hook(self):
        if self._hook is None:
            raise AttributeError('hook is not set')
        return self._hook

    @hook.setter
    def hook(self, val):
        self._hook = val
        self._hook.agent = self
        print(f'from now, {self._hook}')
        # self._hook.add_endpoint(endpoint='/act', handler=self.act)

    def act(self) -> np.ndarray:
        """
        act returns its running env's action space shaped/typed action
        """
        raise NotImplementedError

    def collect(self, s, a, r, d, s_) -> 'Maybe Some statistics?':
        """
        collects state and store in buffer
        """
        raise NotImplementedError

    def train(self) -> 'Maybe Train Result?':
        """
        train it's model by calling model.step num_epochs times
        """
        raise NotImplementedError


class MAgent:
    """
    interface.
    agent has buffer
    even pg algorhtms use buffer(not the true experience replay buffer but it triggers train when buffer is full)
    agent counts interactions and do what it needs to do when it needs to be done
    """

    def __init__(
            self,
            # model must be instantiated and initialized before passing as argument
            models: List[Union[TorchModel, TFModel]],
            train_interval,
            num_epochs,
            # device,
            buffer_cls,
            buffer_kwargs,
    ):
        self.curr_step = 0
        self.train_interval = train_interval

        # """
        self.models = models

        self.num_epochs = num_epochs
        # self.device = device

        num_agents = len(models)
        self.buffers = []
        for model in self.models:
            self.buffers.append(buffer_cls(
                size=buffer_kwargs['size'],
                state_shape=model.observation_shape,
                action_shape=model.action_shape
            ))
        self._hook = None

    @property
    def hook(self):
        if self._hook is None:
            raise AttributeError('hook is not set')
        return self._hook

    @hook.setter
    def hook(self, val):
        self._hook = val
        self._hook.agent = self
        print(f'from now, {self._hook}')
        # self._hook.add_endpoint(endpoint='/act', handler=self.act)

    def act(self) -> np.ndarray:
        """
        act returns its running env's action space shaped/typed action
        """
        raise NotImplementedError

    def collect(self, s, a, r, d, s_) -> 'Maybe Some statistics?':
        """
        collects state and store in buffer
        """
        raise NotImplementedError

    def train(self) -> 'Maybe Train Result?':
        """
        train it's model by calling model.step num_epochs times
        """
        raise NotImplementedError
