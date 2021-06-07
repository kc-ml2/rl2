#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json

import argparse
import warnings
import random
import settings

import numpy as np
import torch

from distutils.dir_util import copy_tree

from rl2 import collectors, envs, agents, models, defaults

import rl2.utils.common as common
from rl2.utils.distributions import CategoricalHead, ScalarHead
from rl2.agents.agent import AbstractAgent
from rl2.modules import DeepMindEnc


# In[52]:


from rl2.envs.gym.atari import make_atari


# In[3]:


seed = 42
env = 'atari'
env_id = 'Breakout'


# In[4]:


eps = 1e-8
inf = 1e8


# In[5]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[6]:


env = make_atari(env_id+'NoFrameskip-v4', 1, 42)


# Base class
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


# Your implementation

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

        advs = (advs - advs.mean()) / (advs.std() + settings.EPS)

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

# env & input shape

# In[9]:


# env = getattr(envs, args.env)(args)

# Create network components for the agent
input_shape = env.observation_space.shape


# reshape from tf to pt

# In[10]:


if len(input_shape) > 1:
    input_shape = (input_shape[-1], *input_shape[:-1])


# networks

# In[11]:


encoder = DeepMindEnc(input_shape).to(device)
actor = CategoricalHead(encoder.out_shape, env.action_space.n).to(device)
critic = ScalarHead(encoder.out_shape, 1).to(device)

networks = [encoder, actor, critic]


# In[12]:


# Declare optimizer
optimizer = 'torch.optim.Adam'


# In[47]:


from rl2 import models


# In[48]:


# Create a model using the necessary networks
model = models.ActorCriticModel(networks, optimizer)


# In[42]:


# Create a collector for managing data collection
collector = collectors.PGCollector(env, model, device)


# In[ ]:


agent = PPOAgent(model, collector)


# In[14]:


from tqdm import tqdm


# In[60]:


# # Finally create an agent with the defined components
# train(args, 'PPOAgent', 'ppo', model, collector)
steps = int(5e7)
steps = steps // collector.num_workers + 1
for step in tqdm(range(5)):
#     if train_fn is None:
    agent.train()
#     else:
#         train_fn(agent, step, steps)


# In[79]:


model.__dir__()


# In[61]:


model = agent.model


# In[62]:


model.encoder


# torch.save(model.nets[0].state_dict(), 'deepmindenc.pt')

# torch.save(model.nets[1].state_dict(), 'categoricalhead.pt')

# torch.save(model.nets[2].state_dict(), 'scalarhead.pt')

# In[64]:


model.fo


# In[63]:


torch.jit.save(torch.jit.script(model.nets[0]), 'deepmindenc_script.pt')


# In[29]:


sp = env.observation_space.shape


# In[30]:


input_shape = (sp[-1], *sp[:-1])


# In[32]:


sp


# In[40]:


s = model.nets[0].out_shape


# In[43]:


example_inputs = torch.rand(s).to(device)


# In[48]:


from torch.distributions.categorical import Categorical


# In[49]:


dist = Categorical()


# In[47]:


script = torch.jit.trace(model.nets[1], example_inputs)


# In[35]:


torch.jit.save(script, 'categoricalhead_script.pt')


# In[ ]:


torch.jit.save(torch.jit.script(model.nets[2]), 'scalarhead_script.pt')


# In[108]:


loaded_net1 = torch.jit.load('deepmindenc_script.pt')


# In[109]:


loaded_net2 = torch.jit.load('categoricalhead.pt')


# In[ ]:





# In[102]:


loaded_net.state_dict()


# In[98]:


torch.load('deepmindenc.pt')


# In[81]:


next(net1.modules())


# In[48]:


gen = net1.parameters()


# In[67]:


aa = next(gen)
aa


# In[68]:


aa.shape


# In[84]:


print(torch.save.__doc__)


# In[85]:


net1.state_dict()


# In[ ]:




