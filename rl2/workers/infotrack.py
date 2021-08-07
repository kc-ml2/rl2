import numpy as np
from collections import deque


class InfoTracker(dict):
    def __init__(self, keys, smoothing_keys=[], smoothing_const=10):
        super().__init__()
        # self= {}
        self.smooth_keys = smoothing_keys
        self.smooth_const = smoothing_const
        self.smooth_buffer = {}
        for k in self.smooth_keys:
            self.smooth_buffer[k] = deque(maxlen=smoothing_const)

    def update(self, new_info):
        assert isinstance(new_info, dict)
        for k, v in new_info.items():
            if k in self.smooth_keys:
                self.smooth_buffer[k].append(v)
                self[k] = np.mean(list(self.smooth_buffer[k]))
            else:
                self[k] = v
