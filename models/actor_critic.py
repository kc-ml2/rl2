from models.model import AbstractModel


class ActorCriticModel(AbstractModel):
    def __init__(self, args, encoder, actor, critic, optimizer,
                 optim_args={},
                 **kwargs):
        self.encoder, self.actor, self.critic = encoder, actor, critic

        self.set_optimizer([self.encoder, self.actor, self.critic],
                           optimizer,
                           optim_args)

    def infer(self, x):
        ir = self.encoder(x)
        ac_dist = self.actor(ir)
        val_dist = self.critic(ir)
        return ac_dist, val_dist
