import torch.nn as nn

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
