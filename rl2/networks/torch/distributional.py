import torch
from torch import nn

from rl2.distributions.torch import ScalarDist, SampleDist, CategoricalDist, DiagGaussianDist, MixtureGaussianDist
import torch.nn.functional as F

"""
some neural net components that outputs distributions
"""


class ScalarHead(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.linear = nn.Linear(input_size, out_size)

    def forward(self, x) -> "torch.distrib":
        x = self.linear(x)
        dist = ScalarDist(x)
        return dist


class SampleHead(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        assert out_size == 1
        # Only handles univariate cases
        self.linear = nn.Linear(input_size, out_size)

    def forward(self, x):
        x = self.linear(x).squeeze(-1)
        dist = SampleDist(x)
        return dist


class CategoricalHead(nn.Module):
    def __init__(self, input_size, out_size):
        super().__init__()
        self.linear = nn.Linear(input_size, out_size)

    def forward(self, x):
        x = self.linear(x)
        dist = CategoricalDist(logits=F.log_softmax(x, dim=-1))
        return dist


class DiagGaussianHead(nn.Module):
    def __init__(self, input_size, out_size, min_var=0.0, lim=None):
        super().__init__()
        self.out_size = out_size
        self.linear = nn.Linear(input_size, out_size * 2)
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


class MixtureGaussianHead(nn.Module):
    def __init__(self, input_size, out_size, n_mix=5, min_var=0):
        super().__init__()
        self.out_size = out_size
        self.n_mix = n_mix
        self.min_var = min_var
        self.linear1 = nn.Linear(input_size, out_size * self.n_mix)
        self.linear2 = nn.Linear(input_size, out_size * self.n_mix)
        self.linear3 = nn.Linear(input_size, out_size * self.n_mix)

    def forward(self, x):
        mus = self.linear1(x)
        sigs = torch.sqrt(F.softplus(self.linear2(x)) + self.min_var)
        ws = torch.exp(F.log_softmax(self.linear3(x), dim=-1))
        dist = MixtureGaussianDist(mus, sigs, ws)
        return dist
