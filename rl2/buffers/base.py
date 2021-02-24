import random
from typing import List
from collections import Iterable
import numpy as np
import warnings


class ReplayBuffer:
    def __init__(self, size, s_shape=(4, 84, 84), decimal=True, more={}):
        # FIXME: Mutable default argument
        max_size = int(size)
        self.max_size = max_size
        data_type = np.float32 if decimal else np.uint8
        self.data_type = np.float32 if decimal else np.uint8
        self.s = np.ones((max_size, *s_shape), dtype=np.uint8)
        self.a = np.ones((max_size,), dtype=np.uint8)
        # FIXME: AttributeError: 'ReplayBuffer' object has no attribute 'data_type'
        # self.r = np.zeros((max_size,), dtype=self.data_type)
        self.r = np.zeros((max_size,), dtype=data_type)
        self.d = np.ones((max_size,), dtype=np.uint8)
        self.s_ = np.ones((max_size, *s_shape), dtype=np.uint8)
        self.more = more.keys()
        for k, shape in more.items():
            assert type(shape) == tuple
            setattr(self, k, np.ones((max_size, *shape), dtype=self.data_type))

        self.curr_idx = 0
        self.curr_size = 0

    def __getitem__(self, sample_idx):
        return [self.s[sample_idx], self.a[sample_idx], self.r[sample_idx], self.d[sample_idx], self.s_[sample_idx]]

    def __setattr__(self, key, value):
        # FIXME: 'ReplayBuffer' object has no attribute 'max_size'
        # if self.max_size != len(value):
        #     raise ValueError(
        #         f'buffer max size != {key} length, {self.max_size} != {len(value)}')
        pass

    def to_dict(self):
        d = {
            'state': self.s,
            'action': self.a,
            'reward': self.r,
            'done': self.d,
            'next_state': self.s_,
        }

        return d

    def to_df(self):
        try:
            import pandas as pd
            df = pd.DataFrame(self.to_dict(), copy=True)
            return df
        except:
            warnings.warn('pandas not installed')
            pass

    @property
    def is_full(self):
        return (self.curr_size == self.max_size)

    # def _ordered(func):
    #     def decorator(self, *args, **kwargs):
    #         self.curr_size = min(self.curr_size + 1, self.max_size)
    #         self.ins_idx = self.curr_idx
    #
    #         result = func(self, *args, **kwargs)
    #
    #         self.curr_idx = (self.curr_idx + 1) % self.max_size
    #         return result
    #
    #     return decorator
    #
    # def _random(func):
    #     def decorator(self, *args, **kwargs):
    #         self.curr_size = min(self.curr_size + 1, self.max_size)
    #         if self.curr_idx >= self.max_size:
    #             self.ins_idx = np.random.randint(self.max_size)
    #         else:
    #             self.ins_idx = self.curr_idx
    #
    #         result = func(self, *args, **kwargs)
    #
    #         if self.curr_idx < self.max_size:
    #             self.curr_idx += 1
    #         return result
    #
    #     return decorator

    # @_ordered
    def push(self, s, a, r, d, s_, *args, **kwargs):
        self.curr_size = min(self.curr_size + 1, self.max_size)
        # if self.curr_idx >= self.max_size:
        #     self.ins_idx = np.random.randint(self.max_size)
        # else:
        #     self.ins_idx = self.curr_idx

        # insert index
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
        # if self.curr_idx < self.max_size:
        #     self.curr_idx += 1
        # return self.curr_size

    def sample(self, num, idx=None) -> List["ReplayBuffer"]:
        if idx is None:
            sample_idx = np.random.randint(self.curr_size, size=num)
        else:
            sample_idx = idx
        samples = [self[sample_idx]]
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

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory[self.position] = transition
        if len(self.memory) < self.capacity:
            self.position += 1
        else:
            self.position = random.randrange(self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
