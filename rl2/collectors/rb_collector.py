import numpy as np
import torch
from collections import deque
from .collector import GeneralCollector
from rl2.modules import ReplayBuffer


class RBCollector(GeneralCollector):
    def __init__(self, args, env, model):
        super().__init__(args)
        self.args = args
        self.env = env
        self.model = model
        self.remain = 0

        self.obs = self.env.reset()
        self.epinfobuf = deque(maxlen=100)
        self.reset_buffer()

        self.frames = 0

    def has_next(self):
        return bool(self.remain)

    def reset_buffer(self):
        self.buffer = ReplayBuffer(self.args.rb_size,
                                   s_shape=self.env.observation_space.shape)

    def reset_count(self):
        self.remain = 1

    def step_env(self, logger=None, info=None):
        for step in range(self.args.n_step):
            if logger is not None and info is not None:
                self.log(logger, info)
            if self.frames % self.args.target_update < self.args.num_workers:
                self.model.update_target()

            self.collect(info)
            self.frames += self.args.num_workers

        while self.buffer.curr_size < self.args.init_collect:
            self.collect(info)
            self.frames += self.args.num_workers

    def step(self):
        samples = self.buffer.sample(self.args.batch_size)
        obs, acs, rews, dones, obs_, _ = map(
            lambda x: torch.FloatTensor(x).to(self.args.device),
            samples
        )
        if len(obs.shape) == 4:
            obs = obs.permute(0, 3, 1, 2)
            obs_ = obs_.permute(0, 3, 1, 2)
        self.remain -= 1
        return obs, acs, rews, dones, obs_

    def collect(self, info=None, **kwargs):
        self.eps = max(1.0 - self.frames / (self.args.steps * 0.1),
                       0.0) * 0.99 + 0.01
        obs = self.obs.copy()
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.FloatTensor(obs).to(self.args.device)
        with torch.no_grad():
            q_dist = self.model.infer(obs)
            q_val = q_dist.mean
            acs = q_val.argmax(-1)
        obs, acs = map(lambda x: x.cpu().numpy(), [obs, acs])
        rand_acs = np.random.randint(self.env.action_space.n, size=acs.shape)
        rand_mask = (np.random.rand(*acs.shape) < self.eps).astype(np.long)
        acs = acs + rand_mask * (rand_acs - acs)
        obs_, rews, dones, infos = self.env.step(acs)

        obs_ = obs_.astype(np.uint8)
        acs = acs.astype(np.uint8)
        rews = rews.astype(np.float32)
        for o, a, r, d, o_ in zip(self.obs, acs, rews, dones, obs_):
            self.buffer.push(o, a, r, d, o_)
        self.obs = obs_

        if info is not None:
            info.update('Values/Reward', rews.mean().item())
        for info in infos:
            if 'episode' in info:
                self.epinfobuf.append(info['episode']['score'])
        return q_dist
