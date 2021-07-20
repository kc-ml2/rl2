from typing import Union, List, Tuple

from rl2.models.base import TorchModel
from rl2.models.tf.base import TFModel


class Agent:
    """
    interface.
    agent has buffer
    even pg algorhtms use buffer(not the true experience replay buffer but it triggers train when buffer is full)
    agent counts interactions and do what it needs to do when it needs to be done
    """

    def __init__(
            self,
            model: Union[TorchModel, TFModel],
            buffer_cls=None,
            buffer_kwargs={},
            num_epochs: int = 0,
            num_envs: int = 1,
            train_interval: int = 128,
            eval_interval: int = 0
    ):
        self.model = model

        self.curr_step = 0
        self.tasks: List[Tuple[callable, callable]] = []

        self.train_interval = train_interval
        self.train_at = lambda x: x % self.train_interval == 0
        self.eval_interval = eval_interval
        self.eval_at = lambda x: x % self.eval_interval == 0

        self.num_epochs = num_epochs
        if self.train_interval > 0:
            self.buffer = buffer_cls(**buffer_kwargs)

        self.num_envs = num_envs
        if self.num_envs == 1:
            self.done = False
        else:
            self.done = [False] * num_envs
        self.obs = None

        if self.model.recurrent:
            self.model._init_hidden(self.done)
            self.prev_hidden = self.hidden = self.model.hidden

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

    def act(self):
        """
        act returns its running env's action space shaped/typed action
        """
        raise NotImplementedError

    def collect(self, state, action, reward, done, next_state):
        """
        collects state and store in buffer
        """
        raise NotImplementedError

    def train(self):
        """
        train it's model by calling model.step num_epochs times
        """
        raise NotImplementedError

    def step(self):
        """
        result = {}
        for exec_at, task in self.tasks:
            if exec_at(self.curr_step):
                task_result = task()
                result = {**result, **task_result}

        return result
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
            models: List[Union[TorchModel, TFModel]],
            train_interval,
            num_epochs,
            buffer_cls,
            buffer_kwargs,
    ):
        self.curr_step = 0
        self.train_interval = train_interval

        self.models = models

        self.num_epochs = num_epochs

        num_agents = len(models)
        self.buffers = []
        for model in self.models:
            buffer = buffer_cls(
                size=buffer_kwargs['size'],
                state_shape=model.observation_shape,
                action_shape=model.action_shape
            )
            self.buffers.append(buffer)
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

    def act(self):
        """
        act returns its running env's action space shaped/typed action
        """
        raise NotImplementedError

    def collect(self, state, action, reward, done, next_state):
        """
        collects state and store in buffer
        """
        raise NotImplementedError

    def train(self):
        """
        train it's model by calling model.step num_epochs times
        """
        raise NotImplementedError
