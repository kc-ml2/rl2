import numpy as np


def general_advantage_estimation(trajectories: dict, value_p, dones,
                                 gamma, lamda):
    # General Advantage Estimation
    gae = 0
    advs = np.zeros(trajectories['value'].shape[0], 1)
    n_step = advs.shape[0]
    done_mask = 1 - dones

    for t in reversed(range(n_step)):
        if t != n_step - 1:
            value_p = trajectories['value'][t + 1]
            done_mask = 1.0 - trajectories['done'][t + 1].float()
        rews = trajectories['reward'][t]

        while len(done_mask.shape) < len(value_p.shape):
            done_mask = np.expand_dims(done_mask, axis=1)
        while len(rews.shape) < len(value_p.shape):
            rews = np.expand_dims(rews, axis=1)

        value = trajectories['value'][t]
        delta = rews + done_mask * gamma * value_p - value
        gae = delta + done_mask * gamma * lamda * gae
        advs[t] = gae

    return advs.detach()