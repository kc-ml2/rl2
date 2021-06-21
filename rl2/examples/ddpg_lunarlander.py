import csv
from torch.utils.tensorboard import SummaryWriter
from termcolor import colored
import logging
import traceback
from datetime import datetime
from pathlib import Path
import sys
import gym
import os
from easydict import EasyDict
from rl2.agents.configs import DEFAULT_DDPG_CONFIG
from rl2.agents.ddpg import DDPGAgent, DDPGModel
from rl2.workers.base import EpisodicWorker
from torch.utils.tensorboard.writer import SummaryWriter

"""
you might want to modify
1. layer architecture -> just pass nn.Module to predefined models
2. which distributions to use -> implement model from interfaces e.g. implement ActorCritic for custom PPO
3. how to sample distributions -> customize Agent
etc...

below example just changes 1. and some hparams
"""

# FIXME: Temp; remove later

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

    def scalar_summary(self, info, step, lvl="INFO", tag='values'):
        assert isinstance(info, dict), "data must be a dictionary"
        # flush to terminal
        # if self.args.log_level <= LOG_LEVELS[lvl]['lvl']:
        if self.args.log_level <= 10000:
            key2str = {}
            for key, val in info.items():
                if isinstance(val, float):
                    valstr = "%-8.3g" % (val,)
                else:
                    valstr = str(val)
                key2str[self._truncate(key)] = self._truncate(valstr)

            if len(key2str) == 0:
                self.log("empty key-value dict", 'WARNING')
                return

            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

            dashes = '  ' + '-'*(keywidth + valwidth + 7)
            lines = [dashes]
            for key, val in key2str.items():
                lines.append('  | %s%s | %s%s |' % (
                    key,
                    ' '*(keywidth - len(key)),
                    val,
                    ' '*(valwidth - len(val))
                ))
            lines.append(dashes)
            print('\n'.join(lines))

        # flush to csv
        if self.log_dir is not None:
            filepath = Path(os.path.join(self.log_dir, tag + '.csv'))
            if not filepath.is_file():
                with open(filepath, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step'] + list(info.keys()))

            with open(filepath, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([step] + list(info.values()))

        # flush to tensorboard
        if self.writer is not None:
            for k, v in info.items():
                self.writer.add_scalar(k, v, step)

    def add_histogram(self, tag, values, step):
        if self.writer is not None:
            self.writer.add_histogram(tag, values, global_step=step)

    def add_hparams(self, hparams, metrics):
        if self.writer is not None:
            self.writer.add_hparams(hparams, metrics)

    def _truncate(self, s):
        return s[:20] + '...' if len(s) > 23 else s


env = gym.make('MountainCarContinuous-v0')
# env = gym.make('LunarLanderContinuous-v2')


config = DEFAULT_DDPG_CONFIG

myconfig = {
    'num_workers': 64,
    'buffer_size': int(1e5),
    'batch_size': 128,
    'num_epochs': 1,
    'update_interval': 1,
    'train_interval': 1,
    'log_interval': 10,
    'optim_ac': 'torch.optim.Adam',  # 'RMSprop'
    'optim_cr': 'torch.optim.Adam',
    'lr_ac': 1e-4,
    'lr_cr': 1e-3,
    'gamma': 0.99,
    'eps': 0.00001,
    'polyak': 0.99,
    'grad_clip': 0.01,
    'loss_fn': 'mse_loss',  # 'smoothl1loss'
    'log_dir': './runs',
    'tag': '',
    'log_level': 10
}

config = EasyDict(myconfig)

# writer = SummaryWriter()
if __name__ == '__main__':
    logger = Logger(name='DEFAULT', args=config)
    # hparams = dict(config)
    # logger.add_hparams(hparams, {})
    observation_shape = env.observation_space.shape
    action_shape = (env.action_space.n,) if hasattr(
        env.action_space, 'n') else env.action_space.shape

    model = DDPGModel(observation_shape=observation_shape,
                      action_shape=action_shape,
                      optim_ac=config.optim_ac,
                      optim_cr=config.optim_cr,
                      lr_ac=config.lr_ac,
                      lr_cr=config.lr_cr,
                      grad_clip=config.grad_clip,
                      polyak=config.polyak,
                      is_save=True)

    agent = DDPGAgent(model,
                      action_low=env.action_space.low,
                      action_high=env.action_space.high,
                      update_interval=config.update_interval,
                      train_interval=config.train_interval,
                      num_epochs=config.num_epochs,
                      buffer_size=config.buffer_size,
                      eps=config.eps,
                      gamma=config.gamma,
                      log_interval=config.log_interval,
                      logger=logger)

    worker = EpisodicWorker(env=env,
                            num_envs=1,
                            agent=agent,
                            training=True,
                            max_episodes=1e4,
                            max_steps_per_ep=1e3,
                            log_interval=config.log_interval,
                            render=False,
                            logger=logger,
                            config=config)

    worker.run()
