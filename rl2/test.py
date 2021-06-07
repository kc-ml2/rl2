#!/usr/bin/env python
# coding: utf-8

import sys

import torch

sys.path.insert(0, '.')
from . import collectors, models
from utils.distributions import CategoricalHead, ScalarHead

from .agents.agent import AbstractAgent
from .modules import DeepMindEnc


# In[2]:


from rl2.envs.gym.atari import make_atari


# # choose your env

# In[3]:


seed = 42
env = 'atari'
env_id = 'Breakout'


# # constants

# In[4]:


eps = 1e-8
inf = 1e8


# In[5]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# # make env

# In[6]:


env = make_atari(env_id+'NoFrameskip-v4', 1, 42)


# # Base class
# 
# inherit this class to implement your agent

# In[7]:


class GeneralAgent(AbstractAgent):
    name = 'General'

    def __init__(self, model, collector, epoch):
#         self.args = args
        self.model = model
        self.epoch = epoch
        self.collector = collector
#         self.logger = Logger(self.name, args=args)

#         self.info = EvaluationMetrics([
#             'Time/Step',
#             'Time/Item',
#             'Loss/Total',
#             'Loss/Value',
#             'Values/Reward',
#             'Values/Value',
#             'Score/Train',
#         ])

    def loss_func(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        self.collector.step_env()
        for epoch in range(self.epoch):
            self.collector.reset_count()
            while self.collector.has_next():
                data = self.collector.step()
                loss = self.loss_func(*data, info=self.info)
                self.model.step(loss)


# # Your implementation
# 
# example ppo

# In[8]:


class PPOAgent(GeneralAgent):
    name = 'PPO'

    def __init__(self,
                 model,
                 collector,
                 epoch=4,
                 vf_coef=0.5,
                 ent_coef=0.01,
                 cliprange=0.1):
        super().__init__(model=model, collector=collector, epoch=epoch)

        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.cliprange = cliprange

#         self.metrics.set([
#             'Loss/Policy',
#             'Values/Entropy',
#             'Values/Adv',
#         ])

    def loss_func(self, obs, old_acs, old_nlps, advs, old_rets):
        ac_dist, val_dist = self.model.forward(obs)
        vals = val_dist.mean
        nlps = -ac_dist.log_prob(old_acs)
        ent = ac_dist.entropy().mean()
        old_vals = old_rets - advs

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        vals_clipped = (old_vals + torch.clamp(vals - old_vals,
                                               -self.cliprange,
                                               self.cliprange))

        vf_loss_clipped = 0.5 * (vals_clipped - old_rets.detach()).pow(2)
        vf_loss = 0.5 * (vals - old_rets.detach()).pow(2)

        vf_loss = torch.max(vf_loss, vf_loss_clipped).mean()

        ratio = torch.exp(old_nlps - nlps).unsqueeze(-1)
        pg_loss1 = -advs * ratio

        ratio = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)
        pg_loss2 = -advs * ratio

        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Total loss
        loss = pg_loss - self.ent_coef * ent + self.vf_coef * vf_loss

        if self.metrics is not None:
            self.metrics.update('Values/Value', vals.mean().item())
            self.metrics.update('Values/Adv', advs.mean().item())
            self.metrics.update('Values/Entropy', ent.item())
            self.metrics.update('Loss/Value', vf_loss.item())
            self.metrics.update('Loss/Policy', pg_loss.item())
            self.metrics.update('Loss/Total', loss.item())

        return loss


# ppo

# In[9]:


input_shape = env.observation_space.shape


# reshape from tf to pt

# In[10]:


if len(input_shape) > 1:
    input_shape = (input_shape[-1], *input_shape[:-1])


# networks

# In[11]:


# In[12]:


# class CategoricalHead(nn.Module):
#     def __init__(self, input_size, out_size):
#         super().__init__()
#         self.linear = nn.Linear(input_size, out_size)

#     def forward(self, x):
#         x = self.linear(x)
#         dist = Categorical(logits=F.log_softmax(x, dim=-1))
#         return dist


# In[13]:


# class ScalarHead(nn.Module):
#     def __init__(self, input_size, out_size):
#         super().__init__()
#         self.linear = nn.Linear(input_size, out_size)

#     def forward(self, x):
#         x = self.linear(x)
#         dist = ScalarDist(x)
#         return dist


# In[14]:


encoder = DeepMindEnc(input_shape).to(device)
actor = CategoricalHead(encoder.out_shape, env.action_space.n).linear.to(device)
critic = ScalarHead(encoder.out_shape, 1).linear.to(device)

networks = [encoder, actor, critic]


# In[15]:


# Declare optimizer
optimizer = 'torch.optim.Adam'


# In[16]:


# Create a model using the necessary networks
model = models.ActorCriticModel(networks, optimizer)


# In[17]:


# Create a collector for managing data collection
collector = collectors.PGCollector(env, model, device)


# In[18]:


agent = PPOAgent(model, collector)


# In[19]:


from tqdm import tqdm


# In[20]:


# # Finally create an agent with the defined components
# train(args, 'PPOAgent', 'ppo', model, collector)
steps = int(5e7)
steps = steps // collector.num_workers + 1
for step in tqdm(range(5)):
#     if train_fn is None:
    agent.train()
#     else:
#         train_fn(agent, step, steps)


# In[ ]:


model = agent.model


# In[ ]:


input_shape


# In[ ]:


example_inputs = torch.rand((32,*input_shape)).to(device)
example_inputs.shape


# In[ ]:


model


# In[ ]:


#1 torch.save
torch.save(agent.model, 'just_prms.pt')

#ok


# In[ ]:


#2 torch.jit.script
sc = torch.jit.script(model)
torch.jit.save(sc, 'ppo_script.pt')


# In[ ]:


#3 torch.jit.trace
sc = torch.jit.trace(model, example_inputs)
torch.jit.save(sc, 'ppo_trace.pt')


# In[ ]:


#4 torch.jit.trace_module
# sc = torch.jit.trace_module(model, #dict input)
torch.jit.save(sc, 'ppo_trace.pt')