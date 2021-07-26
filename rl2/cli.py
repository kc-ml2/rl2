import json
import logging
import os
import runpy
import sys
from io import StringIO
from pathlib import Path

import click
import gin
import yaml

from rl2.ctx import var

logger = logging.getLogger(__name__)


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
# @click.option('--mode', default='test', type=click.Choice(['train', 'test']))
# @click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda']))
@click.option('--script', type=click.Path())
@click.option('--example', type=str)
@click.option('--config', type=click.Path(exists=True))
@click.option(
    '--log-dir',
    default=os.path.join(Path.home(), 'rl2-runs'),
    type=click.Path()
)
@click.option('--log-level', default=logging.INFO)
def main(script, example, config, log_dir, log_level):
    logger.setLevel(log_level)

    if config is None:
        config = {}
    else:
        config = handle_extension(config)
    config['log_dir'] = log_dir
    var.set(config)

    if script is not None:
        out = runpy.run_path(script, run_name='__main__')
        logger.debug(out)
    elif example is not None:
        # assert example.endswith('.py')
        out = runpy.run_module(example, run_name='__main__')
        logger.debug(out)

    # if mode == 'train':n
    #     pass
