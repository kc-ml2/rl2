from _rl2.models.model import AbstractModel


class ActorCriticModel(nn.Module):
    def __init__(self, args, networks, optimizer,
                 optim_args={},
                 **kwargs):
        assert len(networks) == 3
        super().__init__(networks)
        self.encoder, self.actor, self.critic = tuple(self.nets)

        self.optimizer = self.set_optimizer(
            [self.encoder, self.actor, self.critic],
            optimizer,
            optim_args
        )

        self.max_grad = 0.5

    # def forward(self, x):
    #     ir = self.encoder(x)
    #     ac_dist = self.actor(ir)
    #     val_dist = self.critic(ir)
    #     return ac_dist, val_dist

    def act(self, x):
        ir = self.encoder(x)
        ac_dist = self.actor(ir)
        return ac_dist.sample()
