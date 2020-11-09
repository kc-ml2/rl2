import copy
import torch
import torch.nn as nn
from agents.agent import GeneralAgent


class DQNAgent(GeneralAgent):
    name = 'DQN'

    def __init__(self, args, model, collector):
        super().__init__(args, model, collector)
        self.target_net = copy.deepcopy(self.model)
        self.criterion = nn.SmoothL1Loss()

    def loss_func(self, obs, acs, rews, dones, obs_, info=None):
        q_dist = self.model.infer(obs)
        if len(q_dist.mean.shape) != len(acs.shape):
            acs = acs.unsqueeze(-1)
        q_val = torch.gather(q_dist.mean, -1, acs.long())
        with torch.no_grad():
            q_dist_tar = self.model.infer_target(obs_)
            q_tar = q_dist_tar.mean.max(-1, keepdim=True)[0]
        loss = self.criterion(q_val, q_tar)

        if info is not None:
            info.update('Loss/Total', loss.item())
            info.update('Loss/Value', loss.item())
            info.update('Values/Value', q_val.mean().item())

        return loss
