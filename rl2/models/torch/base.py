import importlib
from types import FunctionType
from typing import Iterable
from abc import abstractmethod
import itertools
import copy

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F

from rl2.networks.torch import MLP, ConvEnc, LSTM
import rl2.distributions.torch as dist

"""
interface that can handle most of the recent algorithms. (PG, Qlearning)
but interface itself can serve as vanilla algorithm
"""


class TorchModel(nn.Module):
    """
    input : state
    output : infered torch tensor or tuple of tensors if multiple.

    this class encapsulates all neural net models
    + other models(e.g. decision tree) of an algorithm
    inherits nn.Module to utilize its functionalities, e.g. model.state_dict()
    child classes must init TorchModel for torch utilities
    """

    def __init__(
            self,
            observation_shape: tuple,
            action_shape: tuple,
            device: str = None,
            **kwargs
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device if device else available_device

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
    def get_optimizer_by_name(modules: Iterable, optim_name: str,
                              **optim_kwargs) -> Optimizer:
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
        device = kwargs.get('device')
        super().__init__(observation_shape, action_shape,
                         device=device)

    @abstractmethod
    def forward(self, state):
        """
        TODO: implement vanila pg learning
        """
        pass

    @abstractmethod
    def step(self, loss):
        """
        TODO:
        implement vanila pg step
        """
        pass

    @abstractmethod
    def save(self):
        """
        TODO:
        implement vanila pg save
        """
        pass

    @abstractmethod
    def load(self):
        """
        implement vanila pg load
        """
        pass


class ValueBasedModel(TorchModel):
    """
    interface for Q learning models.
    Q learning models always have addtional target network

    must implement abstractmethods to inherit this class.
    but this class can be used as vanila Q learning, also.
    """

    def __init__(self, observation_shape, action_shape, **kwargs):
        device = kwargs.get('device')
        super().__init__(observation_shape, action_shape,
                         device)

        self.q_network = None
        self.target_network = None

    def update_trg(self, alpha=0.0):
        self.polyak_update(self.q_network, self.target_network, alpha)

    @abstractmethod
    def forward(self, state):
        """
        TODO: implement vanila q learning
        """
        pass

    @abstractmethod
    def step(self, loss):
        """
        TODO: implement vanila q learning
        """

        pass

    @abstractmethod
    def save(self):
        """
        TODO: implement vanila q learning
        """

        pass

    @abstractmethod
    def load(self):
        """
        TODO: implement vanila q learning
        """
        pass


class BaseEncoder(nn.Module):
    def __init__(self, net, encoded_dim, reorder=False, flatten=False,
                 recurrent=False):
        super().__init__()
        self.net = net
        self.reorder = reorder
        self.flatten = flatten
        self.memory = {}
        self.rnn = None
        if recurrent:
            self.rnn = LSTM(encoded_dim, encoded_dim)

    def forward(self, x, **kwargs):
        if self.rnn:
            # Check that the input data is in form (S, N, F...)
            assert len(x.shape) in (3, 5)
            seq_len = x.shape[0]
            batch_size = x.shape[1]
            x = x.reshape(-1, *x.shape[2:])

        if self.reorder or self.flatten:
            assert len(x.shape) > 3, "Dimension of input too small for encoder"
        input_id = id(x)
        if input_id in self.memory:
            return self.memory[input_id]

        if self.net is not None:
            if self.reorder:
                x = x.permute(0, 3, 1, 2)
            x = self.net(x)
        elif self.flatten:
            x = x.view(x.shape[0], -1)

        if self.rnn:
            x = x.reshape(seq_len, batch_size, *x.shape[1:])
            x, hidden = self.rnn(x, **kwargs)
            x = x.reshape(-1, *x.shape[2:])
            self.memory[input_id] = x
            return x, hidden
        else:
            self.memory[input_id] = x
            return x

    def reset_memory(self):
        self.memory = {}


class BaseHead(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        if self.net is not None:
            x = self.net(x)
        return x


class BranchModel(TorchModel):
    def __init__(self,
                 observation_shape,
                 action_shape,
                 encoder=None,
                 encoded_dim=64,
                 head=None,
                 optimizer='torch.optim.Adam',
                 lr=1e-4,
                 grad_clip=1.0,
                 make_target=False,
                 discrete=True,
                 deterministic=True,
                 reorder=False,
                 flatten=False,
                 default=True,
                 head_depth=1,
                 recurrent=False,
                 **kwargs):
        device = kwargs.get('device')
        super().__init__(observation_shape, action_shape,
                         device)
        optim_args = kwargs.get('optim_args', {})
        high = kwargs.get('high', 1)

        self.discrete = discrete
        self.deterministic = deterministic
        self.grad_clip = grad_clip
        self.recurrent = recurrent

        if (not encoder and not default) or flatten:
            # Observation space will be the encoded space
            encoded_dim = sum(list(observation_shape))
        self.encoder = self._handle_encoder(
            encoder, observation_shape, encoded_dim,
            reorder=reorder, flatten=flatten, default=default, high=high,
            rnn=recurrent
        ).to(self.device)

        self.head = self._handle_head(
            head, action_shape, encoded_dim, discrete, deterministic,
            depth=head_depth,
        ).to(self.device)

        self.optimizer = self.get_optimizer_by_name(
            [self.encoder, self.head], optimizer, **optim_args
        )
        self.encoder_target = None
        self.head_target = None
        if make_target:
            self.encoder_target = copy.deepcopy(self.encoder)
            self.head_target = copy.deepcopy(self.head)

    def _handle_encoder(self, encoder, observation_shape, encoded_dim,
                        reorder=False, flatten=False, default=True, rnn=False,
                        high=1):
        if encoder:
            # User has given the encoder, validate!
            if reorder and len(observation_shape) < 2:
                raise ValueError("Cannont reorder a input dimension < 2")
            return BaseEncoder(encoder, encoded_dim, reorder, False, rnn)
        if len(observation_shape) == 3:
            if flatten:
                return BaseEncoder(encoder, encoded_dim, False, flatten, rnn)
            else:
                # Build a default convnet
                if reorder:
                    observation_shape = (
                        observation_shape[2],
                        observation_shape[0],
                        observation_shape[1],
                    )
                encoder = ConvEnc(observation_shape, encoded_dim, high=high)
                return BaseEncoder(encoder, encoded_dim, reorder, flatten, rnn)
        elif len(observation_shape) == 1:
            # Make MLP
            if default:
                encoder = nn.Sequential(MLP(observation_shape[0], encoded_dim),
                                        nn.ReLU())
            return BaseEncoder(encoder, encoded_dim, False, False, rnn)
        else:
            raise ValueError("Cannot create a default encoder for "
                             "observation of dimension > 3 or 2")

    def _handle_head(self, head, action_shape, encoded_dim,
                     discrete, deterministic, depth=0):
        # User may have given the head module.
        # Validate that head is compatible with the parameterization
        # to be used.
        output_dim = action_shape[0]
        dims = [encoded_dim, output_dim]
        if discrete:
            if deterministic:
                # distribution = 'GumbelSoftmax'
                # Change for dqn default head
                # remove GumbelSoftmax if DPG cant handle discrete action_space
                distribution = 'Scalar'
            else:
                distribution = 'Categorical'
        else:
            if deterministic:
                distribution = 'Scalar'
            else:
                distribution = 'Gaussian'
                # TODO: Add other distributions
                # ex) GMM, quantile, beta, etc.
        head = getattr(dist, distribution + 'Head')(*dims, module=head)

        return head

    def reset_encoder_memory(self):
        for mod in self.children():
            if isinstance(mod, BaseEncoder):
                mod.reset_memory()

    def update_trg(self, alpha=0.0):
        self.polyak_update(self.encoder, self.encoder_target, alpha)
        self.polyak_update(self.head, self.head_target, alpha)

    def forward(self, observation, **kwargs):
        if len(observation.shape) == len(self.observation_shape):
            observation = observation.unsqueeze(0)
        if (self.recurrent
           and len(observation.shape) == len(self.observation_shape) + 1):
            observation = observation.unsqueeze(0)
        ir = self.encoder(observation, **kwargs)
        if self.recurrent:
            hidden = ir[1]
            ir = ir[0]
            output = self.head(ir)
            return output, hidden

        output = self.head(ir)
        return output

    def forward_trg(self, observation, **kwargs):
        with torch.no_grad():
            if len(observation.shape) == len(self.observation_shape):
                observation = observation.unsqueeze(0)
            ir = self.encoder_target(observation, **kwargs)
            if self.recurrent:
                hidden = ir[1]
                ir = ir[0]
                output = self.head(ir)
                return output, hidden

            output = self.head_target(ir)
        return output

    def step(self, loss, retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), self.grad_clip)
        self.optimizer.step()

    @abstractmethod
    def save(self):
        """
        TODO: implement vanila q learning
        """
        pass

    @abstractmethod
    def load(self):
        """
        TODO: implement vanila q learning
        """
        pass


class InjectiveBranchModel(BranchModel):
    def __init__(self,
                 observation_shape,
                 action_shape,
                 injection_shape,
                 encoder=None,
                 encoded_dim=64,
                 head=None,
                 optimizer='torch.optim.Adam',
                 lr=1e-4,
                 grad_clip=1.0,
                 make_target=False,
                 discrete=True,
                 deterministic=True,
                 default=True,
                 reorder=False,
                 flatten=False,
                 **kwargs):
        super().__init__(
            observation_shape, action_shape, encoder=encoder,
            encoded_dim=encoded_dim, head=head,  optimizer=optimizer, lr=lr,
            grad_clip=grad_clip, make_target=make_target, discrete=discrete,
            deterministic=deterministic, default=default, reorder=reorder,
            flatten=flatten,
            **kwargs)
        optim_args = kwargs.get('optim_args', {})

        if (not encoder and not default) or flatten:
            # Observation space will be the encoded space
            encoded_dim = sum(list(observation_shape))
        encoded_dim += injection_shape[0]
        self.head = self._handle_head(
            head, action_shape, encoded_dim, discrete, deterministic
        ).to(self.device)

        self.optimizer = self.get_optimizer_by_name(
            [self.encoder, self.head], optimizer, **optim_args
        )
        self.head_target = None
        if make_target:
            self.head_target = copy.deepcopy(self.head)

    def forward(self, observation, injection, *args):
        # TODO: check for shared memory of the same trace
        if len(observation.shape) == len(self.observation_shape):
            observation = observation.unsqueeze(0)
        ir = self.encoder(observation, *args)
        ir = torch.cat([ir, injection], dim=-1)
        output = self.head(ir)

        return output

    def forward_trg(self, observation, injection, *args):
        with torch.no_grad():
            # TODO: check for shared memory of the same trace
            if len(observation.shape) == len(self.observation_shape):
                observation = observation.unsqueeze(0)
            ir = self.encoder_target(observation, *args)
            ir = torch.cat([ir, injection], dim=-1)
            output = self.head_target(ir)

        return output
