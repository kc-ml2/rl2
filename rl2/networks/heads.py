import torch
import torch.nn as nn
import torch.nn.functional as F

from rl2.distributions import ScalarDist, SampleDist, GumbelSoftmaxDist, \
    CategoricalDist, DiagGaussianDist, MixtureGaussianDist
from rl2.networks.core import MLP


class HeadModule(nn.Module):
    def __init__(self, input_size, out_size, module, depth):
        super().__init__()
        hidden = [128 for _ in range(depth)]
        if not module:
            self.linear = MLP(input_size, out_size, hidden_dims=hidden)
        else:
            self.linear = module


class ScalarHead(HeadModule):
    def __init__(self, input_size, out_size, module=None, depth=0):
        super().__init__(input_size, out_size, module, depth)

    def forward(self, x):
        x = self.linear(x)
        dist = ScalarDist(x)
        return dist


class SampleHead(HeadModule):
    def __init__(self, input_size, out_size, module=None, depth=0):
        super().__init__(input_size, out_size, module, depth)
        # Only handles univariate cases
        assert out_size == 1

    def forward(self, x):
        x = self.linear(x).squeeze(-1)
        dist = SampleDist(x)
        return dist


class GumbelSoftmaxHead(HeadModule):
    def __init__(self, input_size, out_size, module=None, depth=0):
        super().__init__(input_size, out_size, module, depth)

    def forward(self, x):
        x = self.linear(x)
        dist = GumbelSoftmaxDist(x)
        return dist


class CategoricalHead(HeadModule):
    def __init__(self, input_size, out_size, module=None, depth=0):
        super().__init__(input_size, out_size, module, depth)

    def forward(self, x):
        x = self.linear(x)
        dist = CategoricalDist(logits=F.log_softmax(x, dim=-1))
        return dist


class DiagGaussianHead(HeadModule):
    def __init__(self, input_size, out_size, min_var=0.0, lim=None,
                 module=None, depth=0):
        super().__init__(input_size, out_size, module, depth)
        self.linear = nn.Linear(input_size, out_size * 2)
        self.out_size = out_size
        self.min_var = min_var
        self.lim = False
        if lim is not None:
            self.lim = True
            low = torch.FloatTensor(lim[0]).unsqueeze(0)
            high = torch.FloatTensor(lim[1]).unsqueeze(0)
            self.register_buffer('low', low)
            self.register_buffer('high', high)

        # self.lim = lim

    def forward(self, x):
        x = self.linear(x)
        mu = x[:, :self.out_size]
        if self.lim:
            mu = 0.5 * (F.tanh(mu) + 1.0) / (self.high - self.low) + self.low
        sig = torch.sqrt(F.softplus(x[:, self.out_size:]) + self.min_var)
        # dist = DiagGaussianDist(mu, sig, lim=self.lim)
        dist = DiagGaussianDist(mu, sig, lim=None)
        return dist


class MixtureGaussianHead(HeadModule):
    def __init__(self, input_size, out_size, n_mix=5, min_var=0, module=None,
                 depth=0):
        super().__init__(input_size, out_size, module, depth)
        self.out_size = out_size
        self.n_mix = n_mix
        self.min_var = min_var
        self.linear = nn.Linear(input_size, out_size * self.n_mix)
        self.linear2 = nn.Linear(input_size, out_size * self.n_mix)
        self.linear3 = nn.Linear(input_size, out_size * self.n_mix)

    def forward(self, x):
        mus = self.linear(x)
        sigs = torch.sqrt(F.softplus(self.linear2(x)) + self.min_var)
        ws = torch.exp(F.log_softmax(self.linear3(x), dim=-1))
        dist = MixtureGaussianDist(mus, sigs, ws)
        return dist
