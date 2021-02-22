import torch
from importlib import import_module
from rl2.models.torch.base import PolicyBasedModel, ValueBasedModel
from torch.distributions import Distribution

from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.agents.base import Agent
from rl2.models.torch.base import PolicyBasedModel, ValueBasedModel
# from rl2.models.torch.dpg import DPGModel
# from rl2.models.torch.actor_critic import ActorCriticModel
from rl2.buffers.base import ReplayBuffer
from rl2.networks.torch.networks import MLP
from rl2.networks.torch.distributional import ScalarHead

# from rl2.utils import noise

# class DDPGModel(DPGModel, ActorCriticModel):
class ActorModel(PolicyBasedModel):
    pass

class CriticModel(ValueBasedModel):
    pass

class DDPGModel(PolicyBasedModel, ValueBasedModel):
    """
    predefined model
    (same one as original paper)
    """

    def __init__(self, input_shape, **kwargs):
        super().__init__(input_shape, **kwargs)
        config = kwargs['config']
        self.encoder_ac = MLP(in_shape=input_shape, out_shape=encode_dim)
        self.encoder_cr = MLP(in_shape=(input_shape + action_dim), out_shape=encode_dim)
        self.actor = ScalarHead(input_size=encoder_ac.out_shape, out_size=1)
        self.critic = ScalarHead(input_size=encoder_cr.out_shape, out_size=1)

        nn.Sequential([encoder, actor])
        nn.Sequential([encoder, critic])
        nn
        self.networks = {"enc_ac": self.encoder_ac,
                         "enc_cr": self.encoder_cr,
                         "ac": self.actor,
                         "cr": self.critic}  # For target update

        self.networks['en c_ac']
        self.encoder_ac
        pass
    
    def act(self, obs: "observation") -> Distribution:
        ir = self.encoder_ac(s)
        a_dist = self.actor(ir)
        
        return a_dist
        

        pass
    
    def infer_ac(self, obs) -> Distribution:
        ir = self.encoder_ac(obs)
        ac_dist = self.actor(ir)

        return ac_dist

    def infer_cr(self, obs, act) -> Distribution:
        input = torch.cat[obs, act]
        ir = self.encoder(input)
        val_dist = self.critic(ir)
        return val_dist

    def infer(self, obs) -> Distribution:
        ac_dist = self.infer_ac(obs)
        act = ac_dist.mean
        ir_cr = self.encoder_cr(torch.cat[obs, act])
        val_dist = self.critic(ir_cr)
        
        return ac_dist, val_dist

    def step(self, loss):
        loss_ac, loss_cr = loss

        update ac, cr

        pass

    def save(self):
        # torch.save(os.path.join(save_dir, 'encoder_ac.pt'))
        # torch.save(os.path.join(save_dir, 'encoder_cr.pt'))
        # torch.save(os.path.join(save_dir, 'actor.pt'))
        # torch.save(os.path.join(save_dir, 'critic.pt'))
        pass

    def load(self):
        pass


class DDPGAgent(Agent):
    def __init__(self, model: DDPGModel, **kwargs):
        # config = kwargs['config']
        super().__init__(model, **kwargs)
        self.config = kwargs['config']
        self.buffer = ReplayBuffer()

    def act(self, obs):
        self.model.act(obs)

        return action

    def step(self):
        if self.curr_step % self.update_interval == 0:
            raise NotImplementedError
            
    def train(self):
        trg_ac = copy.deepcopy()
        for i_epoch in range(self.num_epochs):
            data = self.buffer.sample()
            loss = self.loss_func(*data)
            self.model.step(loss)

    def collect(self, s, a, r, d, s_):
        self.curr_step += 1
        self.buffer.push(s, a, r, d, s_)
    
    def loss_func_ac(self, data):
        o = data['obs']
        a = self.model.infer()
        _, val_dist = self.model.infer(o, a)
        q = val_dist.mean()



    def loss_func_cr(self, data):
        o = data['obs']
        q = self.model.infer(o, a)
        self.critic(
        

        pass

    def loss_func(self, data, **kwargs):
        obs = data['obs']
        ac_dist, val_dist = self.model.infer(obs)
        vals = val_dist.mean
        ac = ac_dist.mean

        ac_loss = self.loss_func_ac(**kwargs)
        val_loss = self.loss_func_cr(**kwargs)

        loss = [ac_loss, val_loss]
        raise NotImplementedError

        return loss
