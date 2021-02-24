import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution
from collections.abc import Iterable

from rl2.agents.base import Agent
from rl2.models.torch.base import TorchModel
from rl2.models.torch.ddpg import DDPGModel
from rl2.buffers import ReplayBuffer
from rl2.networks import MLP


def loss_func(data,
              model: TorchModel,
              gamma: int):
    state = data.state
    action = data.action
    done = torch.from_numpy(
    q =
    bellman_target = data.reward + gamma * (1 - data.done) * target_q_prime
    critic_loss = (bellman_target - q)
    return loss


class MADDPGModel(TorchModel):
    def __init__(self, obs_shape, ac_shape, index, full_ac_shape, **kwargs):
        super().__init__()
        self.obs_shape = obs_shape
        self.index = index
        self.lr = kwargs.get('lr', 1e-3)
        if len(obs_shape) > 1:
            assert 'ac_encoder' in kwargs
        emb_shape = 128
        self.ac_encoder = kwargs.get('ac_encoder', MLP(obs_shape, emb_shape))
        self.cr_encoder = kwargs.get('cr_encoder',
                                     MLP(obs_shape + full_ac_shape, emb_shape))
        # TODO: change these to spit out distributions
        self.ac_head = kwargs.get('ac_head', MLP(emb_shape, ac_shape))
        self.cr_head = kwargs.get('cr_head', MLP(emb_shape, 1))

        self.actor = nn.Sequential(self.ac_encoder, self.ac_head)
        self.critic = nn.Sequential(self.cr_encoder, self.cr_head)

        # TODO: get optim string from args
        optim = 'torch.optim.Adam'
        self.actor_optim = self.get_optimizer(self.actor, optim)
        self.critic_optim = self.get_optimizer(self.critic, optim)

    def forward(self, obs, other_acs):
        action = self.actor(obs)
        other_acs.insert(self.index, action)
        other_acs.insert(0, obs)
        critic_input = torch.cat(other_acs, dim=-1)
        q_value = self.critic(critic_input)

        # TODO: these should be distributions?
        return action, q_value

    def step(self, actor_loss, critic_loss):
        self.actor.zero_grad()
        self.critic.zero_grad()
        self.actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        self.critic.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()

    def act(self, obs) -> np.array:
        action = self.actor(obs)
        return action.cpu().numpy()

    def save(self):
        pass

    def load(self):
        pass


class MADDPGAgent(Agent):
    def __init__(self, num_agents, obs_shapes, ac_shapes, **kwargs):
        config = kwargs['config']
        # prioritized experience replay
        # self.per = config.pop('per', False)
        # self.buffer = PrioritizedReplayBuffer() if self.per else ReplayBuffer()

        super().__init__(**kwargs)

        self.models = []
        full_ac_shape = np.asarray(ac_shapes).sum()
        for i in range(num_agents):
            self.models.append(
                MADDPGModel(obs_shapes[i], ac_shapes[i], full_ac_shape, i))

        # TODO: change these to get form kwargs
        start_eps = 0.9
        end_eps = 0.01
        self.eps = start_eps
        self.eps_func = lambda x, y: max(end_eps, x - start_eps / y)
        self.explore_steps = 1e5

        buffer_size = kwargs.get('buffer_size', int(1e5))
        self.batch_size = kwargs.get('buffer_size', 32)
        self.buffers = [ReplayBuffer(buffer_size, s_shape=model.obs_shape)
                        for model in self.models]

    def act(self, obss):
        noise_scale = 0.1  # TODO: change this to attribute
        self.eps = self.eps_func(self.eps, self.explore_steps)
        actions = []
        for model, obs in zip(self.models, obss):
            action = model.act(obs)
            noise = self.eps * np.random.randn_like(action)
            actions.append(action + noise)

        return actions

    def step(self, obss: Iterable, acs: Iterable, rews: Iterable,
             dones: Iterable, obss_p: Iterable):
        self.collect()

        ## Check for update step & update model
        if self.curr_step % self.update_interval == 0:
            self.train()

    def train(self):
        losses = []
        for i, model in enumerate(self.models):
            # TODO: get the buffer size
            transitions = self.buffer[i].sample(self.batch_size)
            loss = self.loss_func(transitions)
        for model, loss in zip(self.models, losses):
            model.step(loss)

    def collect(self, obss: Iterable, acs: Iterable, rews: Iterable,
                dones: Iterable, obss_p: Iterable):
        ## Store given observations
        for i, (obs, ac, rew, done, obs_p) in enumerate(
            zip(obss, acs, rews, dones, obss_p)):
            self.buffers[i].push(obs, ac, rew, done, obs_p)
