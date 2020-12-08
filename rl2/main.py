import os
import json
from distutils.dir_util import copy_tree
import argparse
import warnings
import random
import numpy as np
import torch
import rl2.utils.common as common

from rl2 import collectors, envs, models, settings, defaults
from rl2.modules import DeepMindEnc
from rl2.utils.distributions import CategoricalHead, ScalarHead

warnings.simplefilter('ignore', UserWarning)

def train(args, agent_name, agent_src, model, collector, train_fn=None):
    """Template function for training various agents.
    """
    from rl2 import agents
    # Create agent and save current state
    agent = getattr(agents, agent_name)(args, model, collector)
    logger = agent.logger
    log_dir = logger.log_dir

    path = os.path.join(log_dir, 'config.json')
    logger.log("Saving current arguments to {}".format(path))
    with open(path, 'w') as f:
        json.dump(vars(args), f)

    src = os.path.join(settings.PROJECT_ROOT, 'agents', agent_src)
    dst = os.path.join(log_dir, 'src')
    logger.log("Saving relevant source code to {}".format(dst))
    os.makedirs(dst)
    copy_tree(src, dst)

    # Begin training
    env = "{}:{}".format(args.env, args.env_id)
    logger.log("Begin training {} in ".format(agent_name) + env)
    steps = args.steps // args.num_workers + 1
    for step in range(steps):
        if train_fn is None:
            agent.train()
        else:
            train_fn(agent, step, steps)
    logger.log("Finished training {} in ".format(agent_name) + env)


def a2c(args):
    # Create an environment
    env = getattr(envs, args.env)(args)

    # Create network components for the agent
    input_shape = env.observation_space.shape
    if len(input_shape) > 1:
        input_shape = (input_shape[-1], *input_shape[:-1])
    encoder = DeepMindEnc(input_shape).to(args.device)
    actor = CategoricalHead(encoder.out_shape,
                            env.action_space.n).to(args.device)
    critic = ScalarHead(encoder.out_shape, 1).to(args.device)
    networks = [encoder, actor, critic]
    # Declare optimizer
    optimizer = 'torch.optim.RMSprop'

    # Create a model using the necessary networks
    model = models.ActorCriticModel(args, networks, optimizer)

    # Create a collector for managing data collection
    collector = collectors.PGCollector(args, env, model)

    # Finally create an agent with the defined components
    train(args, 'A2CAgent', 'a2c', model, collector)


def dqn(args):
    # Create an environment
    env = getattr(envs, args.env)(args)

    # Create network components for the agent
    input_shape = env.observation_space.shape
    if len(input_shape) > 1:
        input_shape = (input_shape[-1], *input_shape[:-1])
    encoder = DeepMindEnc(input_shape).to(args.device)
    q_head = ScalarHead(encoder.out_shape, env.action_space.n).to(args.device)
    networks = [encoder, q_head]
    # Declare optimizer
    optimizer = 'torch.optim.RMSprop'

    # Create a model using the necessary networks
    model = models.QvalueModel(args, networks, optimizer)

    # Create a collector for managing data collection
    collector = collectors.RBCollector(args, env, model)

    # Finally create an agent with the defined components
    train(args, 'DQNAgent', 'dqn', model, collector)


def ppo(args):
    # Create an environment
    env = getattr(envs, args.env)(args)

    # Create network components for the agent
    input_shape = env.observation_space.shape
    if len(input_shape) > 1:
        input_shape = (input_shape[-1], *input_shape[:-1])
    encoder = DeepMindEnc(input_shape).to(args.device)
    actor = CategoricalHead(encoder.out_shape,
                            env.action_space.n).to(args.device)
    critic = ScalarHead(encoder.out_shape, 1).to(args.device)
    networks = [encoder, actor, critic]
    # Declare optimizer
    optimizer = 'torch.optim.Adam'

    # Create a model using the necessary networks
    model = models.ActorCriticModel(args, networks, optimizer)

    # Create a collector for managing data collection
    collector = collectors.PGCollector(args, env, model)

    # Finally create an agent with the defined components
    train(args, 'PPOAgent', 'ppo', model, collector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML2 Reinforcement Learning Library"
    )
    parser.add_argument("--load_config", type=str)
    parser.add_argument("--tag", type=str)
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument_group("logger options")
    parser.add_argument("--log_level", type=int, default=20)
    parser.add_argument("--log_step", type=float, default=2e4)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--playback", "-p", action="store_true")
    parser.add_argument("--save_step", type=float, default=None)

    parser.add_argument_group("dataset options")
    parser.add_argument("--env", type=str, default='atari')
    parser.add_argument("--env_id", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_step", type=int, default=128)
    parser.add_argument("--exp", "-e", action="store_true")
    parser.add_argument("--defaults", "-f", action="store_false")
    parser.add_argument("--rb_size", type=int, default=int(1e5))

    parser.add_argument_group("training options")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--steps", type=float, default=5e7)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gam", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--cliprange", type=float, default=0.1)

    args = parser.parse_args()
    if args.load_config is not None:
        path = os.path.join(settings.PROJECT_ROOT, args.load_config)
        with open(path) as config:
            args = common.ArgumentParser(json.load(config))

    if args.defaults:
        config = args.mode
        default_values = getattr(defaults, config)()
        for k, v in default_values.items():
            setattr(args, k, v)
    if args.exp:
        if args.env == 'atari' or args.env == 'bullet':
            config = args.env
            default_values = getattr(defaults, config)()
            for k, v in default_values.items():
                setattr(args, k, v)

    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.tag is None:
        args.tag = args.mode + '/' + args.env_id
    else:
        args.tag = args.mode + '/' + args.env_id + '/' + args.tag
    args.tag = args.tag.lower()

    if args.debug:
        args.log_level = 1
    elif args.quiet:
        args.log_level = 30

    args.steps = int(args.steps)
    args.log_step = int(args.log_step)
    if not hasattr(args, 'device'):
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    globals()[args.mode](args)
