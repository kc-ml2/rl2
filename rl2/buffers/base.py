import random
from typing import Tuple
from collections import Iterable
import numpy as np
import warnings
from collections import namedtuple


class ReplayBuffer:
    def __init__(self, size=1,
                 elements={
                     'state': ((4,), np.float32),
                     'action': ((2,), np.float32),
                     'reward': ((1,), np.float32),
                     'done': ((1,), np.uint8),
                     'state_p': ((4,), np.float32),
                 }):
        self.max_size = int(size)
        if isinstance(elements, dict):
            self.keys = elements.keys()
            self.specs = elements.values()
        elif isinstance(elements, list):
            self.keys = elements
            self.specs = None
        else:
            raise ValueError("Elements can only be a dictionary or a set")
        self.transition = namedtuple("Transition",
                                     ' '.join(list(self.keys)))
        self.transition_idx = namedtuple("Transition",
                                         ' '.join(list(self.keys)) + ' idx')
        self.reset()

    def reset(self):
        if self.specs is not None:
            for k, shape_dtype in zip(self.keys, self.specs):
                if len(shape_dtype) == 2:
                    shape, dtype = shape_dtype
                else:
                    shape = shape_dtype
                    dtype = None
                assert isinstance(shape, Iterable), 'Non-iterable shape given'
                if dtype and dtype.__module__ == 'numpy':
                    setattr(self, k,
                            np.ones((self.max_size, *shape), dtype=dtype))
                else:
                    setattr(self, k, [None] * self.max_size)
        else:
            for k in self.keys:
                setattr(self, k, [None] * self.max_size)

        self.curr_idx = 0
        self.curr_size = 0

    def __getitem__(self, sample_idx):
        items = []
        for key in self.keys:
            item = getattr(self, key)
            if isinstance(item, list):
                item = np.array(item)
            item = item[sample_idx]
            if type(item) == np.ndarray and len(item.shape) < 2:
                item = np.expand_dims(item, axis=-1)
            items.append(item)
        return items

    '''
    def __setattr__(self, key, value):
        if self.max_size != len(value):
            raise ValueError(f'buffer max size != {key} length, '
                             '{self.max_size} != {len(value)}')
    '''

    def to_dict(self):
        d = {}
        for key in self.keys:
            val = getattr(self, key)[:self.curr_size]
            if isinstance(val, list):
                val = np.vstack(val)
            d[key] = val
        return d

    def to_df(self):
        try:
            import pandas as pd
            df = pd.DataFrame(self.to_dict(), copy=True)
            return df
        except ImportError:
            warnings.warn('pandas not installed')

    @property
    def is_full(self):
        return (self.curr_size == self.max_size)

    def push(self, **kwargs):
        for key in self.keys:
            assert key in kwargs.keys()
            getattr(self, key)[self.curr_idx] = kwargs[key]
        self.curr_size = min(self.curr_size + 1, self.max_size)
        self.curr_idx = (self.curr_idx + 1) % self.max_size

    def sample(self, num, idx=None, return_idx=False,
               contiguous=1) -> Tuple[np.ndarray]:
        if idx is None:
            if contiguous > 1:
                num_traj = num // contiguous
                traj_idx = np.random.randint(self.curr_size - num_traj,
                                             size=num_traj)
                sample_idx = (traj_idx.reshape(-1, 1)
                              + np.arange(contiguous).reshape(1, -1))
                sample_idx = sample_idx.flatten()

            else:
                sample_idx = np.random.randint(self.curr_size, size=num)
        else:
            sample_idx = idx
        samples = self[sample_idx]
        if return_idx:
            samples.append(sample_idx)
            return self.transition_idx._make(samples)
        return self.transition._make(samples)
        # return tuple(samples)

    def update_(self, idxs, **new_vals):
        idxs = idxs.astype(np.int32)
        for k, v in new_vals.items():
            getattr(self, k)[idxs] = v


class ExperienceReplay(ReplayBuffer):
    def __init__(self,
                 size=1,
                 state_shape=(1,),
                 action_shape=(1,),
                 state_type=np.float32,
                 action_type=np.float32):
        super().__init__(
            size, elements={
                'state': (state_shape, state_type),
                'action': (action_shape, action_type),
                'reward': ((1,), np.float32),
                'done': ((1,), np.uint8),
                'state_': (state_shape, state_type)
            }
        )

    def reset(self):
        super().reset()

    def push(self, s, a, r, d, s_):
        super().push(state=s, action=a, reward=r, done=d, state_=s_)

    def sample(self, num, idx=None, return_idx=False, contiguous=1):
        transitions = super().sample(num,
                                     idx=idx,
                                     return_idx=return_idx,
                                     contiguous=contiguous)
        output = [transitions.state, transitions.action, transitions.reward,
                  transitions.done, transitions.state_]
        if return_idx:
            output.append(transitions.idx)

        return tuple(output)


class TemporalMemory(ReplayBuffer):
    def __init__(self, size=1, n_env=1):
        super().__init__(
            size, elements=[
                'state',
                'action',
                'reward',
                'done',
                'value',
                'nlp'
            ]
        )
        self.n_env = n_env

    def push(self, s, a, r, d, v, nlp):
        super().push(state=s, action=a, reward=r, done=d, value=v, nlp=nlp)

    def sample(self, num, idx=None, return_idx=False, recurrent=False):
        env_sample_size = 1
        if recurrent:
            assert num > self.n_env
            idx = np.arange(self.max_size)
            env_sample_size = num // self.max_size
        transitions = super().sample(num, idx=idx, return_idx=return_idx)

        if self.n_env > 1:
            if recurrent:
                idx = transitions.idx
                sub_idx = np.random.permutation(self.n_env)[:env_sample_size]
                output = [transitions.state[:, sub_idx],
                          transitions.action[:, sub_idx].reshape(-1, 1),
                          transitions.reward[:, sub_idx].reshape(-1, 1),
                          transitions.done[:, sub_idx],
                          transitions.value[:, sub_idx].reshape(-1, 1),
                          transitions.nlp[:, sub_idx].reshape(-1, 1)]
                mesh = np.array(np.meshgrid(idx, sub_idx)).T.reshape(-1, 2).T
                idx = mesh[0]
                sub_idx = mesh[1]
            else:
                rand_idx = np.random.permutation(num * self.n_env)[:num]
                state = transitions.state
                scalars = [transitions.action, transitions.reward,
                           transitions.done, transitions.value,
                           transitions.nlp]
                output = [state.reshape(-1, *state.shape[2:])[rand_idx],
                          *[el.reshape(-1, 1)[rand_idx] for el in scalars]]
                idx = transitions.idx[rand_idx // self.n_env]
                sub_idx = rand_idx % self.n_env

        # rand_idx = np.random.randint(self.n_env, size=(num, env_sample_size))
        # # state = np.take_along_axis(transitions.state, rand_idx, axis=1)
        # # action = np.take_along_axis(transitions.action, rand_idx, axis=1)
        # state = transitions.state[:, rand_idx]
        # action = transitions.action[:, rand_idx].reshape(-1, 1)
        # reward = transitions.reward[:, rand_idx].reshape(-1, 1)
        # done = transitions.done[:, rand_idx]
        # if not recurrent:
        #     done = done.reshape(-1, 1)
        # value = transitions.value[:, rand_idx].reshape(-1, 1)
        # nlp = transitions.nlp[:, rand_idx].reshape(-1, 1)
        # output = [state, action, reward, done, value, nlp]
        if return_idx:
            output.append((idx, sub_idx))

        return tuple(output)


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
