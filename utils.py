import os
import re
import yaml
import torch.nn as nn

from shutil import copytree
from typing import Callable, Any


class Apply(nn.Module):
    def __init__(self, f: Callable, *args, **kwargs):
        super(Apply, self).__init__()
        self.f = f
        self.kwargs = kwargs
        self.args = args

    def forward(self, input: Any) -> Any:
        kwargs = {}
        args = []
        if not (self.args is None):
            if isinstance(self.args, list):
                if not hasattr(input, '__getitem__') and len(self.args) > 1:
                    raise Exception('The input needs %d components for %s' % (len(self.args), ', '.join(self.args)))
                if len(input) != len(self.args) and len(self.args) > 1:
                    raise Exception('The input needs %d components for %s' % (len(self.args), ', '.join(self.args)))
                for i, arg in self.args:
                    kwargs[arg] = input[i]
        if len(self.args) == 0:
            args = [input]
        kwargs.update(self.kwargs)

        return self.f(*args, **kwargs)

    def __repr__(self) -> str:
        s = 'Apply(%s' % self.f.__name__
        if len(self.args) > 0:
            s += ', '
            s += ', '.join(self.args)
        if len(self.kwargs) > 0:
            s += ', '
            s += ', '.join(['%s=%s' % (k, str(v)) for k, v in self.kwargs.items()])
        s += ')'
        return s


def pprint(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key)+':')
      if hasattr(value, 'items'):
         pprint(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


SPLIT_TOKEN = '.'
VAR_REGEX = '.*\${([a-z]|[A-Z]|_)+([a-z]|[A-Z]|[0-9]|\.|_)*}.*'


# utilities to flatten and re-inflate the configuration for wandb
def _flatten_config(config, prefix, flat_config):
    for key, value in config.items():
        flat_key = SPLIT_TOKEN.join([prefix, key] if prefix else [key])
        if hasattr(value, 'items'):
            _flatten_config(value, flat_key, flat_config)
        elif not isinstance(value, str):
            flat_config[flat_key] = value
        elif not re.match(VAR_REGEX, value):
            flat_config[flat_key] = value


def flatten_config(config):
    flat_config = {}
    _flatten_config(config, None, flat_config)
    return flat_config


def check_config(config, flat_config):
    for key, value in flat_config.items():
        sub_config = config
        keys = key.split(SPLIT_TOKEN)
        for sub_key in keys[:-1]:
            sub_config = sub_config[sub_key]
        if sub_config[keys[-1]] != value:
            print(keys[-1], sub_config[keys[-1]], value)
            raise Exception()


# Add the configuration to the experiment
def add_config(experiment, conf):
    # Create a dictionary with the unresolved configuration
    unresolved_config = dict(yaml.safe_load(OmegaConf.to_yaml(conf, resolve=False)))
    # Ignore some irrelevant configuration
    unresolved_config = {k: v for k, v in unresolved_config.items()}
    # Flatten the configuration
    flat_config = flatten_config(unresolved_config)

    # Update the configuration
    experiment.config.update(flat_config)

    # Check for inconsistencies
    # check_config(conf, wandb.config)

    # Copy hydra config into the files folder so that everything is stored
    copytree('.hydra', os.path.join(experiment.dir, 'hydra'))


