import os
import runpy
from contextvars import ContextVar
from pathlib import Path

import click
import gin
import json
import yaml


def handle_extension(filename):
    if filename.endswith('.json'):
        with open(filename) as fp:
            return json.load(fp)
    elif filename.endswith('.yml'):
        with open(filename) as fp:
            return yaml.load(fp)
    elif filename.endswith('.gin'):
        return gin.parse_config_file(filename)
    else:
        raise NotImplementedError


@click.command()
@click.option('--mode', default='test', type=click.Choice(['train', 'test']))
# @click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']))
@click.option('--config', type=click.Path(exists=True))
@click.option('--log_dir', default='~/rl2-runs', type=click.Path(exists=True))
@click.option('--example', type=str)
@click.option('--script', type=click.File())
def main(mode, config, log_dir, example, script):
    if config is not None:
        config = handle_extension(config)
        config['log_dir'] = log_dir
        var = ContextVar['ctx_config']
        var.set(config)

    if script is not None:
        cwd = Path(__file__).absolute()
        script_path = os.path.join(cwd, script)
        runpy.run_path(script_path)
    elif example is not None:
        # assert example.endswith('.py')
        runpy.run_module(example)

    if mode == 'train':
        pass