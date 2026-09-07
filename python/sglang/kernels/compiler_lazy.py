"""Import-time-free stand-ins for `torch.compiler.disable` /
`torch.compiler.assume_constant_result`.

Evaluating `torch.compiler.<name>` at module import time pulls in torch._dynamo
(~0.6 s) for every process that imports sglang kernels, including ones that
never compile. These wrappers resolve the real decorator on first call.
"""

import functools

import torch


def _lazy(name):
    def decorator(fn):
        wrapped = []

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not wrapped:
                wrapped.append(getattr(torch.compiler, name)(fn))
            return wrapped[0](*args, **kwargs)

        return wrapper

    return decorator


disable = _lazy("disable")
assume_constant_result = _lazy("assume_constant_result")
