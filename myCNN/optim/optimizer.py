import numpy as np

from ..nn.parameter import Parameter

from typing import Iterator


class SGD:
    _param: Iterator[Parameter]

    def __init__(self, lr, momentum=0.9):
        self.lr = lr

    def step(self, param):
        for p in param:
            p.param -= self.lr * p.grad

    def zero_grad(self, param):
        for p in param:
            p.zero_grad()
