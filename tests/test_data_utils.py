import pickle
import numpy as np
from rl2.data_utils import FlatExpertTrajectory
from rl2 import TEST_DATA_DIR

def test_load_data():
    one_hot = np.eye(5)
    expert_trajs = FlatExpertTrajectory(num_episodes=8, one_hot=one_hot)
    expert_trajs.load_pickle(f'{TEST_DATA_DIR}/PPOAgent_trajs.pickle')

def test_set_data():
    one_hot = np.eye(5)
    data_dir = f'{TEST_DATA_DIR}/PPOAgent_trajs.pickle'
    with open(data_dir, 'rb') as fp:
        data = pickle.load(fp)

    expert_trajs = FlatExpertTrajectory(data=data, num_episodes=8, one_hot=one_hot)