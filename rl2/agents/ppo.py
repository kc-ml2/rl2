import torch

from rl2.agents.base import Agent
from rl2.agents.configs import DEFAULT_PPO_CONFIG
from rl2.networks.torch.distributional import CategoricalHead, ScalarHead
from rl2.models.torch.actor_critic import ActorCriticModel
from rl2.networks.torch.networks import DeepMindEnc
from rl2.agents.utils import compute_advantage


class PPOModel(ActorCriticModel):
    """
    predefined model
    (same one as original paper)
    """

    def forward(self, x):
        pass

    def infer(self, x):
        pass

    def __init__(self):
        self.encoder = DeepMindEnc(self.input_shape).to(self.device)
        self.actor = CategoricalHead(self.encoder.out_shape, action_shape).to(self.device)
        self.critic = ScalarHead(self.encoder.out_shape, 1).to(self.device)

    # def forward(self, x):
    #     pass
    #
    # def infer(self, x):
    #     pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

    def step(self, loss):
        pass


class PPOAgent(Agent):
    def __init__(
            self,
            model: ActorCriticModel,
            vf_coef=DEFAULT_PPO_CONFIG['vf_coef'],
            ent_coef=DEFAULT_PPO_CONFIG['ent_coef'],
            clip_param=DEFAULT_PPO_CONFIG['clip_param'],
            **kwargs
    ):
        config = kwargs['config']
        # self.buffer = ReplayBuffer()
        self.model = model
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_param = clip_param

        super().__init__(model, **kwargs)

    def act(self, state):
        ac_dist = self.model.infer(state)
        action = ac_dist.sample()

        return action

    def step(self):
        if self.curr_step % self.update_interval == 0:
            last_obs = self.buffer.o[-1]
            ac_dist, val_dist = self.model(last_obs)
            advs = compute_advantage(self.buffer, val_dist)
            self.buffer.advs = advs
            self.train()
            self.buffer.flush()

    def train(self):
        for i_epoch in range(self.num_epochs):
            data = self.buffer.sample()
            loss = self.loss_func(*data)
            self.model.step(loss)

    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        self.buffer.push(s, a, r, d, s_)

    def loss_func(self, obs, old_acs, old_nlps, advs, old_rets):
        ac_dist, val_dist = self.model.infer(obs)
        vals = val_dist.mean
        nlps = -ac_dist.log_prob(old_acs)
        ent = ac_dist.entropy().mean()
        old_vals = old_rets - advs

        advs = (advs - advs.mean()) / (advs.std() + 1e-7)

        vals_clipped = (old_vals + torch.clamp(vals - old_vals, -self.clip_param, self.clip_param))
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

        # if info is not None:
        #     info.update('Values/Value', vals.mean().item())
        #     info.update('Values/Adv', advs.mean().item())
        #     info.update('Values/Entropy', ent.item())
        #     info.update('Loss/Value', vf_loss.item())
        #     info.update('Loss/Policy', pg_loss.item())
        #     info.update('Loss/Total', loss.item())

        return loss
