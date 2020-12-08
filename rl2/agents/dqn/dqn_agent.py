import copy
import torch
import torch.nn as nn
from rl2.agents import GeneralAgent


class DQNAgent(GeneralAgent):
    name = 'DQN'

    def __init__(self, args, model, collector):
        super().__init__(args, model, collector)
        self.target_net = copy.deepcopy(self.model)
        self.criterion = nn.SmoothL1Loss()
        self.double = True

    def loss_func(self, obs, acs, rews, dones, obs_, info=None):
        q_dist = self.model.infer(obs)
        if len(q_dist.mean.shape) != len(acs.shape):
            acs = acs.unsqueeze(-1)
        q_val = torch.gather(q_dist.mean, -1, acs.long())
        with torch.no_grad():
            q_dist_tar = self.model.infer_target(obs_)
            if self.double:
                q_dist_ = self.model.infer(obs_)
                max_ac = q_dist_.mean.argmax(-1, keepdim=True)
                q_tar = torch.gather(q_dist_tar.mean, -1, max_ac)
            else:
                q_tar = q_dist_tar.mean.max(-1, keepdim=True)[0]
            q_tar *= self.args.gam * (1.0 - dones.unsqueeze(-1))
            q_tar += rews.unsqueeze(-1)
        loss = self.criterion(q_val, q_tar)

        if info is not None:
            info.update('Loss/Total', loss.item())
            info.update('Loss/Value', loss.item())
            info.update('Values/Value', q_val.mean().item())

        return loss
