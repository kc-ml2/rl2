from agents.agent import GeneralAgent


class A2CAgent(GeneralAgent):
    name = 'A2C'

    def loss_func(self, info, obs, old_acs, old_nlps, advs, old_rets):
        ac_dist, val_dist = self.model.infer(obs)
        nlps = -ac_dist.log_prob(old_acs)
        vals = val_dist.mean
        ent = ac_dist.entropy().mean()

        vf_loss = (old_rets.detach() - vals).pow(2).mean()
        pg_loss = (advs.detach() * nlps.unsqueeze(-1)).mean()
        loss = pg_loss - self.args.ent_coef * ent + self.args.vf_coef * vf_loss

        info.update('Values/Value', vals.mean().item())
        info.update('Values/Adv', advs.mean().item())
        info.update('Values/Entropy', ent.item())
        info.update('Loss/Value', vf_loss.item())
        info.update('Loss/Policy', pg_loss.item())
        info.update('Loss/Total', loss.item())

        return loss
