def compute_advantage(...):
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