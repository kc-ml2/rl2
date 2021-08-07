import os
from pathlib import Path

# TODO: change dir to S3 url
TEST_DATA_DIR = os.path.join(Path.home(), 'data')

DEFAULT_FRAMEWORK = 'torch'


def set_default_framework(name):
    global DEFAULT_FRAMEWORK
    DEFAULT_FRAMEWORK = name


def get_global_seed():
    pass


def set_global_seed():
    pass
