import random
import warnings
from collections import Iterable
from collections import namedtuple
from typing import Tuple

import numpy as np

DEFAULT_SIZE = 1024


class ReplayBuffer:
    def __init__(
            self,
            size=DEFAULT_SIZE,
            elements={
                'state': ((4,), np.float32),
                'action': ((2,), np.float32),
                'reward': ((1,), np.float32),
                'done': ((1,), np.uint8),
                'next_state': ((4,), np.float32),
            }
    ):
        self.max_size = int(size)
        if isinstance(elements, dict):
            self.keys = elements.keys()
            self.specs = elements.values()
        elif isinstance(elements, list):
            self.keys = elements
            self.specs = None
        else:
            raise ValueError("Elements can only be a dictionary or a set")

        self.transition = namedtuple(
            "Transition",
            ' '.join(list(self.keys))
        )
        self.transition_idx = namedtuple(
            "Transition",
            ' '.join(list(self.keys)) + ' idx'
        )

        self.reset()

    def _list_type(self):
        pass

    def _dict_type(self):
        pass

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
                    setattr(
                        # np.zeors does not allocate memory
                        self, k, np.ones((self.max_size, *shape), dtype=dtype)
                    )
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

    def to_np(self):
        # tmp
        l = [getattr(self, key)[:self.curr_size] for key in self.keys]
        return np.asarray(l)

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
        return self.curr_size == self.max_size

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
                traj_idx = np.random.randint(self.curr_size - contiguous,
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
    def __init__(
            self,
            size=1,
            state_shape=(1,),
            action_shape=(1,),
            state_type=np.float32,
            action_type=np.float32
    ):
        super().__init__(
            size,
            elements={
                'state': (state_shape, state_type),
                'action': (action_shape, action_type),
                'reward': ((1,), np.float32),
                'done': ((1,), np.uint8),
                'next_state': (state_shape, state_type)
            }
        )

    def reset(self):
        super().reset()

    def push(self, state, action, reward, done, next_state):
        super().push(
            state=state, action=action, reward=reward, done=done,
            next_state=next_state
        )

    def sample(self, num, idx=None, return_idx=False, contiguous=1):
        transitions = super().sample(
            num,
            idx=idx,
            return_idx=return_idx,
            contiguous=contiguous
        )
        if contiguous > 1:
            state = transitions.state.reshape(
                contiguous, -1, *transitions.state.shape[1:])
            next_state = transitions.next_state.reshape(
                contiguous, -1, *transitions.next_state.shape[1:])
            done = transitions.done.reshape(
                contiguous, -1)
        else:
            state = transitions.state
            next_state = transitions.next_state
            done = transitions.done
        output = [state, transitions.action, transitions.reward, done,
                  next_state]
        if return_idx:
            output.append(transitions.idx)

        return tuple(output)


class TemporalMemory(ReplayBuffer):
    def __init__(
            self,
            size: int = DEFAULT_SIZE,
            num_envs: int = 1,
            state_shape=(1,),
            action_shape=(1,),
            state_type=np.float32,
            action_type=np.float32,
    ):
        super().__init__(
            size=size,
            elements=['state', 'action', 'reward', 'done', 'value', 'nlp'],
            # elements={
            #     'state': (state_shape, np.float32),
            #     'action': (action_shape, np.float32),
            #     'reward': ((2,), np.float32),
            #     'done': ((2,), np.uint8),
            #     'value': ((2,), np.float32),
            #     # temporary
            #     'nlp': ((2,), np.float32),
            # }
        )
        self.num_envs = num_envs
        self.shuffle()

    def shuffle(self):
        self.time_idx_queue = np.random.permutation(
            self.max_size * self.num_envs
        )
        # self.time_idx_queue = np.random.permutation(self.max_size)
        self.env_idx_queue = np.random.permutation(self.num_envs)
        self.start = 0

    def push(self, state, action, reward, done, value, nlp):
        super().push(
            state=state,
            action=action,
            reward=reward,
            done=done,
            value=value,
            nlp=nlp
        )

    def sample(self, num, idx=None, return_idx=False, recurrent=False):
        env_sample_size = 1
        num_skip = num
        idx = np.arange(self.max_size)
        if recurrent:
            assert num >= self.max_size
            env_sample_size = num // self.max_size
            num_skip = env_sample_size
        transitions = super().sample(num, idx=idx, return_idx=return_idx)

        # minibatch 뽑는 방법
        # 1. 무조건 random w/ replacement -> randint
        # 2. random w/o replacement -> np.permutation, batchsize 잘라, chunk
        # -> sample_with_replcaeads
        if self.num_envs > 1:
            if recurrent:
                idx = transitions.idx
                # sub_idx = np.random.permutation(self.num_envs)[:env_sample_size]
                sub_idx = self.env_idx_queue[self.start:self.start + num_skip]
                output = [
                    transitions.state[:, sub_idx],
                    transitions.action[:, sub_idx].reshape(-1, 1),
                    transitions.reward[:, sub_idx].reshape(-1, 1),
                    transitions.done[:, sub_idx],
                    transitions.value[:, sub_idx].reshape(-1, 1),
                    transitions.nlp[:, sub_idx].reshape(-1, 1)
                ]
                mesh = np.array(np.meshgrid(idx, sub_idx)).T.reshape(-1, 2).T
                idx = mesh[0]
                sub_idx = mesh[1]
                self.start += num_skip
                if self.start > self.num_envs:
                    self.start = 0
            else:
                # rand_idx = np.random.permutation(self.curr_size * self.num_envs)[:num]
                rand_idx = self.time_idx_queue[self.start: self.start + num_skip]

                state = transitions.state
                scalars = [
                    transitions.action,
                    transitions.reward,
                    transitions.done,
                    transitions.value,
                    transitions.nlp
                ]
                rstate = state.reshape(-1, *state.shape[2:])
                rscalars = [el.reshape(-1, 1)[rand_idx] for el in scalars]
                # state t, e, shape -> t*e, shape
                # assume
                output = [
                    rstate[rand_idx], #
                    *rscalars
                ]

                idx = transitions.idx[rand_idx // self.num_envs]
                sub_idx = rand_idx % self.num_envs
                self.start += num_skip
                if self.start > self.curr_size:
                    self.start = 0
        else:
            raise NotImplementedError

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
