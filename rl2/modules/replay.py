import random
from collections import Iterable
import numpy as np
from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    def __init__(self, size, s_shape=(4, 84, 84), decimal=True, more={}):
        max_size = int(size)
        self.data_type = np.float32 if decimal else np.uint8
        self.s = np.ones((max_size, *s_shape), dtype=np.uint8)
        self.a = np.ones((max_size,), dtype=np.uint8)
        self.r = np.zeros((max_size,), dtype=self.data_type)
        self.d = np.ones((max_size,), dtype=np.uint8)
        self.s_ = np.ones((max_size, *s_shape), dtype=np.uint8)
        self.more = more.keys()
        for k, shape in more.items():
            assert type(shape) == tuple
            setattr(self, k, np.ones((max_size, *shape), dtype=self.data_type))

        self.curr_idx = 0
        self.max_size = max_size
        self.curr_size = 0

    def _ordered(func):
        def decorator(self, *args, **kwargs):
            self.curr_size = min(self.curr_size + 1, self.max_size)
            self.ins_idx = self.curr_idx

            result = func(self, *args, **kwargs)

            self.curr_idx = (self.curr_idx + 1) % self.max_size
            return result
        return decorator

    def _random(func):
        def decorator(self, *args, **kwargs):
            self.curr_size = min(self.curr_size + 1, self.max_size)
            if self.curr_idx >= self.max_size:
                self.ins_idx = np.random.randint(self.max_size)
            else:
                self.ins_idx = self.curr_idx

            result = func(self, *args, **kwargs)

            if self.curr_idx < self.max_size:
                self.curr_idx += 1
            return result
        return decorator

    # @_ordered
    def push(self, s, a, r, d, s_, *args, **kwargs):
        self.curr_size = min(self.curr_size + 1, self.max_size)
        self.ins_idx = self.curr_idx

        self.s[self.ins_idx] = s
        self.a[self.ins_idx] = a
        self.r[self.ins_idx] = r
        self.d[self.ins_idx] = d
        self.s_[self.ins_idx] = s_

        for key in kwargs.keys():
            if key in self.more:
                getattr(self, key)[self.ins_idx] = kwargs[key]

        self.curr_idx = (self.curr_idx + 1) % self.max_size
        return self.curr_size

    def sample(self, num, idx=None):
        if idx is None:
            sample_idx = np.random.randint(self.curr_size, size=num)
        else:
            sample_idx = idx
        samples = [self.s[sample_idx], self.a[sample_idx], self.r[sample_idx],
                   self.d[sample_idx], self.s_[sample_idx]]
        for key in self.more:
            samples.append(getattr(self, key)[sample_idx])
        samples.append(sample_idx)
        return samples

    def update_(self, idxs, **new_vals):
        idxs = idxs.astype(np.int32)
        for k, v in new_vals.items():
            getattr(self, k)[idxs] = v


class ReplayBuffer_:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {}
        self.position = 0

    def push(self, transition):
        self.memory[self.position] = transition
        if len(self.memory) < self.capacity:
            self.position += 1
        else:
            self.position = random.randrange(self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, alpha=0.5, **kwargs):
        super().__init__(capacity, **kwargs)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def push(self, *args):
        super().push(*args)
        idx = self.ins_idx
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.curr_size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0.4):
        assert beta > 0
        idxes = self._sample_proportional(batch_size)
        samples, weights = [], []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.curr_size) ** (-beta)

        samples = super().sample(batch_size, idx=idxes)
        for idx in idxes:
            # samples.append(self.memory[idx])
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.curr_size) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        return samples, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.curr_size
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class EpisodicReplayBuffer:
    def __init__(self, capacity, num_workers, max_length):
        self.num_episodes = capacity // max_length
        self.max_length = max_length
        self.memory = {}
        self.position = 0
        self.trajectory = {i: [] for i in range(num_workers)}

    def push_transition(self, idx, transition):
        self.trajectory[idx].append(transition)
        if len(self.trajectory[idx]) == self.max_length:
            trajectory = np.asarray(self.trajectory[idx])
            self.memory[self.position] = trajectory
            self.trajectory[idx] = []

            if len(self.memory) < self.num_episodes:
                self.position += 1
            else:
                self.position = random.randrange(self.num_episodes)

    def push(self, transitions):
        if not isinstance(transitions, Iterable):
            transitions = [transitions]
        for idx, transition in enumerate(transitions):
            self.push_transition(idx, transition)

    def sample(self, batch_size):
        assert len(self.memory) >= batch_size
        idx = random.sample(self.memory.keys(), batch_size)
        trajs = np.asarray([self.memory[i] for i in idx])
        data = np.transpose(trajs, (1, 0, 2))
        return data

    def __len__(self):
        return len(self.memory)
