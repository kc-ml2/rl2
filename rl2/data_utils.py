import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ExpertTrajectory(Dataset):
    def __init__(
            self,
            # expects list of trajectory of (state, action)
            data: List[List[Tuple[np.ndarray, np.ndarray]]] = None,
            num_episodes: int = None,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            one_hot: np.ndarray = None,
    ):
        self.num_episodes = num_episodes
        self.device = device
        self.one_hot = one_hot

        self.flatten = False
        if len(one_hot.shape) == 2:
            self.flatten = True

        self._data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        if self.num_episodes is not None:
            episodes = np.random.randint(0, len(val), size=self.num_episodes)
            val = [val[i] for i in episodes]

        _data = self._embed_concat(val)
        data = torch.from_numpy(_data)
        self._data = data.to(self.device)
        self.labels = torch.ones(len(self.data)).to(self.device)

    def _embed_concat(self, data):
        _data = []
        for traj in data:
            for state, action in traj:
                action_embedding = self.one_hot[action]
                if self.flatten:
                    state = np.asarray(state).flatten()

                state_action = [
                    state,
                    action_embedding,
                ]
                state_action = np.concatenate(state_action, -1)
                _data.append(state_action)
        _data = np.asarray(_data).astype(np.float32)

        return _data

    def load_pickle(self, data_dir):
        with open(data_dir, 'rb') as fp:
            self.data = pickle.load(fp)
            # type check?




def flatten_concat(states, actions, one_hot):
    if not isinstance(states, np.ndarray) or not isinstance(actions,
                                                            np.ndarray):
        states = np.asarray(states)
        actions = np.asarray(actions)
        rows, cols = states.shape[0], states.shape[1]
        states = states.reshape(rows * cols, -1)
        actions = actions.reshape(rows * cols, -1)

    states = [state.flatten().astype(np.float32) for state in states]
    actions = [one_hot[int(action[0])] for action in actions]
    ret = np.asarray(
        [np.concatenate([i, j]) for i, j in zip(states, actions)])
    ret = torch.from_numpy(ret).float()

    return ret
