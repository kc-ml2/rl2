import math

import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import rl2.ctx

EPS = 1e-8


class ScalarDist:
# class ScalarDist(torch.distributions.Distribution):
    def __init__(self, vals):
        # super().__init__(validate_args=False)
        self.vals = vals

    def __repr__(self):
        return str(self.vals)

    def log_prob(self, x):
        raise NotImplementedError

    @property
    def mean(self):
        return self.vals


class GumbelSoftmaxDist(torch.distributions.Distribution):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def log_prob(self, x):
        raise NotImplementedError

    @property
    def mean(self):
        # return F.softmax(self.logits, dim=-1)
        return F.gumbel_softmax(self.logits, dim=-1, tau=1.0, hard=False)

    def sample(self):
        return F.gumbel_softmax(self.logits, dim=-1, tau=1.0, hard=True)


def MixedDist(main_dist, aux_dists):
    setattr(main_dist, 'auxs', aux_dists)
    return main_dist


class SampleDist(torch.distributions.Distribution):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def log_prob(self, x):
        raise NotImplementedError

    def entropy(self):
        ent = HNA(self.samples)
        return ent

    @property
    def mean(self):
        return self.samples


class CategoricalDist(torch.distributions.Categorical):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def log_prob(self, x, *args, **kwargs):
        return super().log_prob(x.long(), *args, **kwargs)

    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)
        a1 = other.logits - torch.max(other.logits, dim=-1, keepdim=True)
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        z1 = torch.sum(ea1, dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (a0 - torch.log(z0)
                               - a1 + torch.log(z1)), dim=-1)


class DiagGaussianDist(torch.distributions.Normal):
    def __init__(self, loc, scale, lim=None):
        super().__init__(loc, scale)
        self.lim = False
        if lim is not None:
            self.lim = True
            self.low = torch.FloatTensor(lim[0]).to(self.loc)
            self.high = torch.FloatTensor(lim[1]).to(self.loc)

    def rsample(self, n_sample=None, clip=False):
        if n_sample is not None:
            noise = torch.randn(*self.loc.shape, n_sample).to(self.loc)
            if clip:
                noise = torch.clamp(noise, -3.0, 3.0)
            sample = self.loc.unsqueeze(-1) + noise * self.scale.unsqueeze(-1)
            if self.lim:
                sample = torch.min(sample,
                                   self.high.unsqueeze(0).unsqueeze(-1))
                sample = torch.max(sample, self.low.unsqueeze(0).unsqueeze(-1))
        else:
            noise = torch.randn(*self.loc.shape).to(self.loc)
            if clip:
                noise = torch.clamp(noise, -3.0, 3.0)
            sample = self.loc + noise * self.scale
            if self.lim:
                sample = torch.min(sample, self.high.unsqueeze(0))
                sample = torch.max(sample, self.low.unsqueeze(0))
        return sample

    def sample(self, n_sample=None, clip=False):
        return self.rsample(n_sample=n_sample, clip=clip).detach()

    # def log_prob(self, x):
    #     return super().log_prob(x).sum(-1)

    def _log_prob(self, x):
        log_scale = torch.log(self.scale + EPS)
        return -((x - self.loc) ** 2) / (2 * self.var) - log_scale \
               - math.log(math.sqrt(2 * math.pi) + EPS)

    def log_prob(self, x):
        return self._log_prob(x).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1, keepdim=True)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale ** 2

    @property
    def logstd(self):
        return torch.log(self.scale)

    def kl(self, other):
        ss = (self.scale ** 2 + (self.loc - other.loc) ** 2) / (2.0 * self.var)
        return torch.sum(
            other.logstd - self.logstd + ss - 0.5,
            dim=-1
        )


# TODO: implement mixture of gaussian
class MixtureGaussianDist(torch.distributions.Normal):
    def __init__(self, loc, scale, w):
        super().__init__(loc, scale)
        self.w = w

    def sample_param(self, n_sample):
        i_dist = Categorical(probs=self.w)
        i_sample = i_dist.sample((n_sample,)).T
        mu_sample = torch.gather(self.loc, -1, i_sample)
        sig_sample = torch.gather(self.scale, -1, i_sample)
        return mu_sample, sig_sample

    # TODO: implement rsample to be used in entropy
    def sample(self, n_sample=1, clip=False):
        mu_sample, sig_sample = self.sample_param(n_sample)
        noise = torch.randn_like(mu_sample)
        if clip:
            noise = torch.clamp(noise, -3.0, 3.0)
        samples = mu_sample + noise * sig_sample
        return samples.detach()

    def rsample(self, n_sample=1):
        samples = self.sample(n_sample)
        loc = self.loc.unsqueeze(1)
        scale = self.scale.unsqueeze(1)
        noise = (loc - samples.unsqueeze(2)) / scale
        noised_loc = (loc + noise.detach() * scale)
        rsamples = (noised_loc * self.w.unsqueeze(1)).sum(-1)

        return rsamples

    def log_prob(self, x):
        logprob = super().log_prob(x) * self.w
        return logprob.sum(-1)

    def entropy(self):
        # TODO: find a good estimate for entorpy of gaussian mixture
        # return a sample based entorpy estimation
        samples = self.sample(n_sample=64)
        ent = HNA(samples)
        return ent

    @property
    def mean(self):
        return (self.loc * self.w).sum(dim=-1, keepdim=True)


def HNA(samples, m=15, tr=None):
    # Noughabi and Arghami sample based entropy estimation
    # Reference from
    # https://www.sciencedirect.com/science/article/pii/S0377042713006006
    samples = torch.sort(samples, -1)[0]
    if tr is not None:
        assert tr.size(0) == samples.size(0)
        tr = tr.expand_as(samples)
        samples = torch.max(samples, tr)
    n = samples.size(-1)
    summ = 0
    for i in range(n):
        c = 1 if i < m - 1 or i > n - m - 1 else 2
        diff = (samples[..., min(i + m, n - 1)]
                - samples[..., max(i - m, 0)])
        diff = torch.clamp(diff, min=1e-8)
        summ += (1 / n) * torch.log(n / (c * m) * diff)
    return summ.unsqueeze(-1)
