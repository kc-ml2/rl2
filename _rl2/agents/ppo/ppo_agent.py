import torch

from _rl2 import settings
from _rl2.agents import GeneralAgent


class PPOAgent(GeneralAgent):
    name = 'PPO'

    def __init__(self, args, model, collector):
        super().__init__(args, model, collector)
        self.vf_coef = args.vf_coef
        self.ent_coef = args.ent_coef

        # self.max_grad
        self.clip_param = args.cliprange

        self.info.set([
            'Loss/Policy',
            'Values/Entropy',
            'Values/Adv',
        ])

    # obs, acs, nlps, advs, rets
    def loss_func(self, obs, old_acs, old_nlps, advs, old_rets, info=None):
        ac_dist, val_dist = self.model.infer(obs)
        vals = val_dist.mean
        nlps = -ac_dist.log_prob(old_acs)
        ent = ac_dist.entropy().mean()
        old_vals = old_rets - advs

        advs = (advs - advs.mean()) / (advs.std() + settings.EPS)

        vals_clipped = (old_vals + torch.clamp(vals - old_vals,
                                               -self.clip_param,
                                               self.clip_param))
        vf_loss_clipped = 0.5 * (vals_clipped - old_rets.detach()).pow(2)
        vf_loss = 0.5 * (vals - old_rets.detach()).pow(2)
        vf_loss = torch.max(vf_loss, vf_loss_clipped).mean()

        ratio = torch.exp(old_nlps - nlps).unsqueeze(-1)
        pg_loss1 = -advs * ratio
        ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
        pg_loss2 = -advs * ratio
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Total loss
        loss = pg_loss - self.ent_coef * ent + self.vf_coef * vf_loss

        if info is not None:
            info.update('Values/Value', vals.mean().item())
            info.update('Values/Adv', advs.mean().item())
            info.update('Values/Entropy', ent.item())
            info.update('Loss/Value', vf_loss.item())
            info.update('Loss/Policy', pg_loss.item())
            info.update('Loss/Total', loss.item())

        return loss