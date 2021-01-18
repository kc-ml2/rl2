import numpy as np
import torch
from collections import deque
from rl2.collectors.collector import GeneralCollector


class PGCollector(GeneralCollector):
    def __init__(self, args, env, model):
        super().__init__(args)
        self.args = args
        self.env = env
        self.model = model
        self.remain = 0

        self.obs = self.env.reset()
        multi = hasattr(self.env, 'num_envs')
        self.dones = [False] * self.env.num_envs if multi else False
        self.keys = ['obs', 'acs', 'vals', 'nlps', 'rews', 'dones']
        self.epinfobuf = deque(maxlen=100)
        self.reset_buffer()

        # Variables for logging
        self.frames = 0

    def has_next(self):
        return bool(self.remain)

    def reset_buffer(self):
        # gc collect? might memory leak
        self.start = 0
        self.buffer = {k: [] for k in self.keys}

    def reset_count(self):
        self.start = 0
        self.remain = len(self.buffer['advs']) // self.args.batch_size

    def step_env(self, logger=None, info=None):
        self.reset_buffer()
        for step in range(self.args.n_step):
            if logger is not None and info is not None:
                self.log(logger, info)

            self.collect(info)
            self.frames += self.args.num_workers

        for k, v in self.buffer.items():
            v = torch.from_numpy(np.asarray(v))
            self.buffer[k] = v.float().to(self.args.device)
        self.buffer['advs'] = self.compute_advantage()
        for k, v in self.buffer.items():
            self.buffer[k] = v.view(-1, *v.shape[2:])
        self.idx = np.arange(len(self.buffer['advs']))
        self.reset_count()

    def step(self):
        if self.start == 0:
            np.random.shuffle(self.idx)
        end = self.start + self.args.batch_size
        idx = np.array(self.idx[self.start:end])
        obs = self.buffer['obs'][idx]
        acs = self.buffer['acs'][idx]
        nlps = self.buffer['nlps'][idx]
        advs = self.buffer['advs'][idx]
        rets = self.buffer['vals'][idx] + advs

        self.remain -= 1
        self.start = end

        return obs, acs, nlps, advs, rets

    def collect(self, info=None, **kwargs):
        obs = self.obs.copy()
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.FloatTensor(obs).to(self.args.device)
        with torch.no_grad():
            ac_dist, val_dist = self.model.infer(obs)
            acs = ac_dist.sample()
            nlps = -ac_dist.log_prob(acs)
            vals = val_dist.mean
            ents = ac_dist.entropy().squeeze()
        obs, acs, vals, nlps, ents = map(
            lambda x: x.cpu().numpy(),
            [obs, acs, vals, nlps, ents]
        )
        dones = np.asarray(self.dones, dtype=np.float)

        self.buffer['obs'].append(obs)
        self.buffer['acs'].append(acs)
        self.buffer['vals'].append(vals)
        self.buffer['nlps'].append(nlps)
        self.buffer['dones'].append(dones)

        self.obs, rews, self.dones, infos = self.env.step(acs)

        self.dones = self.dones.astype(np.float)
        if info is not None:
            info.update('Values/Reward', rews.mean().item())
        self.buffer['rews'].append(rews)
        for info in infos:
            if 'episode' in info:
                self.epinfobuf.append(info['episode']['score'])
        return ac_dist, val_dist

    def compute_advantage(self, **kwargs):
        # General Advantage Estimation
        gae = 0
        advs = torch.zeros_like(self.buffer['vals'])
        dones = torch.from_numpy(self.dones).to(self.args.device)

        # Look one more frame further
        obs = self.obs.copy()
        if len(obs.shape) == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.FloatTensor(obs).to(self.args.device)
        with torch.no_grad():
            _, val_dist = self.model.infer(obs)
            vals = val_dist.mean

        for t in reversed(range(self.args.n_step)):
            if t == self.args.n_step - 1:
                _vals = vals
                _nonterminal = 1.0 - dones.float()
            else:
                _vals = self.buffer['vals'][t + 1]
                _nonterminal = 1.0 - self.buffer['dones'][t + 1].float()

            while len(_nonterminal.shape) < len(_vals.shape):
                _nonterminal = _nonterminal.unsqueeze(1)
            rews = self.buffer['rews'][t]
            while len(rews.shape) < len(_vals.shape):
                rews = rews.unsqueeze(1)

            vals = self.buffer['vals'][t]
            delta = rews + _nonterminal * self.args.gam * _vals - vals
            gae = delta + _nonterminal * self.args.gam * self.args.lam * gae
            advs[t] = gae

        return advs.detach()
