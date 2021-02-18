import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_shape, out_shape, hidden=[128], activ='ReLU'):
        super().__init__()
        mod_list = []
        prev = in_shape
        for h in hidden:
            mod_list.append(nn.Linear(prev, h))
            mod_list.append(getattr(nn, activ)())
            prev = h
        mod_list.append(nn.Linear(prev, out_shape))
        self.body = nn.Sequential(*mod_list)

    def forward(self, x):
        return self.body(x)


class DeepMindEnc(nn.Module):
    def __init__(self, input_shape, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU()
        )
        sample = torch.FloatTensor(1, *input_shape)
        sample = self.feature(sample)
        sample = sample.view(sample.size(0), -1)
        self.out_shape = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(sample.size(-1), hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x / 255.0
        x = self.feature(x)
        x = x.reshape(x.size(0), -1)
        out = self.fc(x)
        return out


class NoisyLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        p, q = input_dim, output_dim
        self.mu_w = nn.Parameter(torch.rand(p, q) * 0.01)
        self.sig_w = nn.Parameter(torch.ones(p, q) * 0.017)
        self.mu_b = nn.Parameter(torch.zeros(q))
        self.sig_b = nn.Parameter(torch.ones(q) * 0.017)
        self.perturb()

    def perturb(self):
        self.eps_w = nn.Parameter(
            torch.randn_like(self.mu_w),
            requires_grad=False
        )
        self.eps_b = nn.Parameter(
            torch.randn_like(self.mu_b),
            requires_grad=False
        )

    def forward(self, x):
        W = self.mu_w + self.sig_w * self.eps_w
        b = self.mu_b + self.sig_b * self.eps_b
        return x @ W + b.unsqueeze(0)


class NoisyDeepMindEnc(nn.Module):
    def __init__(self, input_shape, hidden_dim=256):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU()
        )
        sample = torch.FloatTensor(1, *input_shape)
        sample = self.feature(sample)
        sample = sample.view(sample.size(0), -1)
        self.fc = nn.Sequential(
            NoisyLinear(sample.size(-1), hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        out = x / 255.0
        out = self.feature(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out