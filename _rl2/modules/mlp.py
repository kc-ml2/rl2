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
