import os
from pathlib import Path
from contextvars import ContextVar
default_config = {
    # if default log_dir changes in cli.py, then change here, too
    'log_dir': os.path.join(Path.home(), 'rl2-runs')
}
var = ContextVar('config', default=default_config)