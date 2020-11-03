import importlib
from models.model import AbstractModel


class ActorCriticModel(AbstractModel):
    def __init__(self, args, encoder, actor, critic, optimizer,
                 optim_args={},
                 **kwargs):
        self.encoder, self.actor, self.critic = encoder, actor, critic
        params = (list(self.encoder.parameters())
                  + list(self.actor.parameters())
                  + list(self.critic.parameters()))
        mod = importlib.import_module('.'.join(optimizer.split('.')[:-1]))
        pkg = optimizer.split('.')[-1]
        self.optimizer = getattr(mod, pkg)(
            params,
            **optim_args
        )

    def infer(self, x):
        ir = self.encoder(x)
        ac_dist = self.actor(ir)
        val_dist = self.critic(ir)
        return ac_dist, val_dist
