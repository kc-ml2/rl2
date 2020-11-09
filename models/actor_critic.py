from models.model import AbstractModel


class ActorCriticModel(AbstractModel):
    def __init__(self, args, networks, optimizer,
                 optim_args={},
                 **kwargs):
        assert len(networks) == 3
        self.encoder, self.actor, self.critic = tuple(networks)

        self.set_optimizer([self.encoder, self.actor, self.critic],
                           optimizer,
                           optim_args)

    def infer(self, x):
        ir = self.encoder(x)
        ac_dist = self.actor(ir)
        val_dist = self.critic(ir)
        return ac_dist, val_dist
