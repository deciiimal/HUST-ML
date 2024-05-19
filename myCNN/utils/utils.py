from typing import Iterator

import numpy as np

from ..nn.parameter import Parameter

def save(path: str, param: Iterator[Parameter]):
    with open(path, 'wb') as f:
        data = [p.param for p in param]
        np.savez(f, *data)


def load(path: str, param: Iterator[Parameter]):
    with np.load(path, allow_pickle=True) as data:
        for p, d in zip(param, data.values()):
            p.param = d
