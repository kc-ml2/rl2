import csv
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict
import numpy as np
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, Popen

import torch
from torch.utils.tensorboard.summary import scalar

from rl2.agents.configs import DEFAULT_DQN_CONFIG
from tensorboard.compat.proto.summary_pb2 import Summary
from termcolor import colored
from torch.utils.tensorboard.writer import FileWriter, SummaryWriter

# Logging levels
LOG_LEVELS = {
    'DEBUG': {'lvl': 10, 'color': 'cyan'},
    'INFO': {'lvl': 20, 'color': 'white'},
    'WARNING': {'lvl': 30, 'color': 'yellow'},
    'ERROR': {'lvl': 40, 'color': 'red'},
    'CRITICAL': {'lvl': 50, 'color': 'red'},
}


class Logger:
    def __init__(self, name, args=None, log_dir=None):
        self.args = args
        if not hasattr(args, 'log_dir'):
            setattr(args, 'log_dir', './runs')
        if not hasattr(args, 'tag'):
            setattr(args, 'tag', '')
        if not hasattr(args, 'log_level'):
            setattr(args, 'log_level', 10)
        if log_dir is None:
            self.log_dir = os.path.join(args.log_dir, args.tag,
                                        datetime.now().strftime("%Y%m%d%H%M%S"))
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = log_dir

        logger = logging.getLogger(name)
        if not logger.handlers:
            format = logging.Formatter(
                "[%(name)s|%(levelname)s] %(asctime)s > %(message)s"
            )
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(format)
            logger.addHandler(streamHandler)
            logger.setLevel(args.log_level)

            filename = os.path.join(self.log_dir, name + '.txt')
            fileHandler = logging.FileHandler(filename, mode="w")
            fileHandler.setFormatter(format)
            logger.addHandler(fileHandler)

        self.logger = logger
        self.writer = SummaryWriter(self.log_dir)
        sys.excepthook = self.excepthook
        self.config_summary(args)
        self.buffer = []

    def log(self, msg, lvl="INFO"):
        lvl, color = self.get_level_color(lvl)
        self.logger.log(lvl, colored(msg, color))

    def add_level(self, name, lvl, color='white'):
        if name not in LOG_LEVELS.keys() and lvl not in LOG_LEVELS.values():
            LOG_LEVELS[name] = {'lvl': lvl, 'color': color}
            logging.addLevelName(lvl, name)
        else:
            raise AssertionError("log level already exists")

    def get_level_color(self, lvl):
        assert isinstance(lvl, str)
        lvl_num = LOG_LEVELS[lvl]['lvl']
        color = LOG_LEVELS[lvl]['color']
        return lvl_num, color

    def excepthook(self, type_, value_, traceback_):
        e = "{}: {}".format(type_.__name__, value_)
        tb = "".join(traceback.format_exception(type_, value_, traceback_))
        self.log(e, "ERROR")
        self.log(tb, "DEBUG")

    def config_summary(self, config):
        with open(self.log_dir+'/config.json', 'w') as f:
            json.dump(config, f)

    def scalar_summary(self, info, step, lvl="INFO", tag='values'):
        """
        info should be dictionary with
        key: str
        value: scalar or dictionary
        if value is dict, it will be used as tag_scalar_dict for add_scalars 
        function and plotted on the same graph.
        """
        assert isinstance(info, dict), "data must be a dictionary"
        # flush to terminal
        if self.args.log_level <= LOG_LEVELS[lvl]['lvl']:
            key2str = {}
            scalars = {}
            for key, val in info.items():
                if isinstance(val, float):
                    valstr = "%-8.3g" % (val,)
                elif isinstance(val, dict):
                    mean = np.mean(list(val.values()))
                    valstr = "%-8.3g" % (mean,)
                    scalars.update({key: val})
                    key = 'Mean/' + key
                else:
                    valstr = str(val)
                key2str[key] = valstr

            if len(key2str) == 0:
                self.log("empty key-value dict", 'WARNING')
                return

            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

            dashes = '  ' + '-'*(keywidth + valwidth + 7)
            lines = [dashes]
            for key, val in sorted(key2str.items()):
                lines.append('  | %s%s | %s%s |' % (
                    key,
                    ' '*(keywidth - len(key)),
                    val,
                    ' '*(valwidth - len(val))
                ))
            lines.append(dashes)
            print('\n'.join(lines))

        # flush to csv
        to_csv = {}
        for key, val in info.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    to_csv.update({key+'/'+k: v})
            else:
                to_csv.update({key: val})
        if self.log_dir is not None:
            filepath = Path(os.path.join(self.log_dir, tag + '.csv'))
            if not filepath.is_file():
                with open(filepath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step'] + list(to_csv.keys()))

            with open(filepath, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([step] + list(to_csv.values()))

        for k, v in scalars.items():
            info.update({k: np.mean(list(v.values()))})

        # flush to tensorboard
        if self.writer is not None:
            for k, v in info.items():
                self.writer.add_scalar(k, v, step)
            for k, v in scalars.items():
                self.add_scalars(k, v, step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_scalars")
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self.writer._get_file_writer().get_logdir()
        for tag, scalar_value in tag_scalar_dict.items():
            fw_tag = fw_logdir + "/" + tag
            if fw_tag in self.writer.all_writers.keys():
                fw = self.writer.all_writers[fw_tag]
            else:
                fw = FileWriter(fw_tag, self.writer.max_queue,
                                self.writer.flush_secs,
                                self.writer.filename_suffix)
                self.writer.all_writers[fw_tag] = fw
            fw.add_summary(scalar(main_tag, scalar_value),
                           global_step, walltime)

    def store_rgb(self, rgb_array):
        # FIXME: safe way to store image buffer
        # self.buffer.push(rgb_array)
        rgb_array = np.transpose(rgb_array, (2, 0, 1))
        self.buffer.append(rgb_array)

    def video_summary(self, tag, step):
        # _, t, h, w, c = self.buffer.shape
        vid_tensor = torch.from_numpy(np.array(self.buffer)).unsqueeze(0)
        if self.writer is not None:
            self.writer.add_video(tag, vid_tensor, step)
        self.buffer = []

    def add_histogram(self, tag, values, step):
        if self.writer is not None:
            self.writer.add_histogram(tag, values, global_step=step)

    def add_hparams(self, hparams, metrics):
        if self.writer is not None:
            self.writer.add_hparams(hparams, metrics)

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s
