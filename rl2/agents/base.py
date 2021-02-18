from typing import Union

import torch

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
            update_interval,
            observation_shape,
            action_shape,
            num_epochs,
            device,
            buffer_cls,
            buffer_kwargs,
    ):
        self.curr_step = 0
        self.update_interval = update_interval

        # """
        self.model = model
        # self.model.input_shape = observation_shape
        self.observation_shape = observation_shape
        self.action_space = action_shape

        self.num_epochs = num_epochs
        self.device = device

        self.buffer = buffer_cls(**buffer_kwargs)

    def act(self) -> Action:
        """
        act returns its running env's action space shaped/typed action
        """
        raise NotImplementedError

    def collect(self) -> Maybe Some statistics?:
        """
        collects state and store in buffer
        """
        raise NotImplementedError

    def train(self) -> Maybe Train Result?:
        """
        train it's model by calling model.step num_epochs times
        """
        raise NotImplementedError

