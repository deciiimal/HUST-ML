import numpy as np

from abc import abstractmethod, ABCMeta
from typing import List, Iterable, Iterator, Optional

from .parameter import Parameter


class Module(metaclass=ABCMeta):
    r"""
    基类
    """
    name = ''
    _modules: List['Module']
    _parameters: List[Parameter]

    def __init__(self, *args, **kwargs) -> None:
        self._modules = []
        ...

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        ...

    def parameters(self) -> Iterator[Parameter]:
        for module in self.modules():
            for val in vars(module).values():
                if isinstance(val, Parameter):
                    yield val

    def modules(self) -> Iterator['Module']:
        if len(self._modules) == 0:
            yield self
        for module in self._modules:
            if module is None:
                continue
            yield from module.modules()

    def train(self):
        ...

    def eval(self):
        ...


class Linear(Module):
    name = 'Linear'

    def __init__(self, in_size, out_size, *, bias=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = Parameter(np.random.normal(0, 0.1, size=(in_size, out_size))
                                .astype(np.float64), 'weight')
        if bias:
            self.bias = Parameter(np.random.normal(0, 0.1, size=(1, out_size))
                                  .astype(np.float64), 'bias')
        else:
            self.bias = Parameter(None, 'bias')

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert input.shape[1] == self.in_size
        setattr(self, 'input', input)
        if self.bias.param is not None:
            return input @ self.weight.param + self.bias.param
        else:
            return input @ self.weight.param

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert hasattr(self, 'input')
        self.weight.grad += self.input.T @ grad
        if self.bias.param is not None:
            self.bias.grad += grad.sum(axis=0)
        return grad @ self.weight.param.T

    def __repr__(self) -> str:
        ...


class Conv2d(Module):
    name = 'Conv2d'

    def __init__(self, in_ch: int, out_ch: int, ksize: int, stride: int, padding: int, *, bias=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ksize = ksize
        self.stride = stride
        # TODO: 需要处理padding是0的情况
        self.padding = padding
        self.weight = Parameter(np.random.normal(
            0, 0.1, size=(in_ch, ksize, ksize, out_ch)).astype(np.float64), 'weight')
        if bias:
            self.bias = Parameter(np.random.normal(
                0, 0.1, size=(1, out_ch, 1, 1)).astype(np.float64), 'bias')
        else:
            self.bias = Parameter(None, 'bias')

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        :return: [N, ch, H, W]
        """
        height = input.shape[2] + 2 * self.padding
        width = input.shape[3] + 2 * self.padding
        if self.padding > 0:
            input_pad = np.zeros((input.shape[0], input.shape[1], height, width), dtype=input.dtype)
            input_pad[:, :, self.padding:-self.padding, self.padding:-self.padding] = input
        else:
            input_pad = input
        out_h = (height - self.ksize) // self.stride + 1
        out_w = (width - self.ksize) // self.stride + 1

        stride_shape = (
            input_pad.shape[0], input_pad.shape[1],
            out_h, out_w,
            self.ksize, self.ksize
        )
        stride_strides = (
            input_pad.strides[0], input_pad.strides[1],
            input_pad.strides[2] * self.stride,
            input_pad.strides[3] * self.stride,
            input_pad.strides[2], input_pad.strides[3]
        )
        setattr(self, 'input_pad', input_pad)
        setattr(self, 'stride_strides', stride_strides)
        setattr(self, 'stride_shape', stride_shape)
        input_strides = np.lib.stride_tricks.as_strided(
            input_pad, shape=stride_shape, strides=stride_strides
        )
        output = np.einsum(
            'nchwxy,cxyo->nohw',
            input_strides, self.weight.param,
            optimize=True)
        if self.bias.param is not None:
            output += self.bias.param
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert hasattr(self, 'input_pad')
        input_strides = np.lib.stride_tricks.as_strided(
            self.input_pad, shape=self.stride_shape, strides=self.stride_strides
        )
        self.weight.grad += np.tensordot(input_strides, grad, axes=((0, 2, 3), (0, 2, 3)))
        if self.bias.param is not None:
            self.bias.grad += np.sum(grad, axis=(0, 2, 3))[np.newaxis, :, np.newaxis, np.newaxis]
        out_grad = np.zeros_like(self.input_pad)
        for ih in range(grad.shape[2]):
            for iw in range(grad.shape[3]):
                out_grad[
                    :, :,
                    ih * self.stride: ih * self.stride + self.ksize,
                    iw * self.stride: iw * self.stride + self.ksize
                ] += np.tensordot(grad[:, :, ih, iw], self.weight.param, axes=((1,), (3,)))
        if self.padding == 0:
            return out_grad
        return out_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]


class AvgPool2d(Module):
    name = 'AvgPool2d'

    def __init__(self, ksize: int, stride: int, padding: int):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.padding = padding

    def forward(self, input: np.ndarray) -> np.ndarray:
        height = input.shape[2] + 2 * self.padding
        width = input.shape[3] + 2 * self.padding
        if self.padding > 0:
            input_pad = np.zeros((input.shape[0], input.shape[1], height, width), dtype=input.dtype)
            input_pad[:, :, self.padding:-self.padding, self.padding:-self.padding] = input
        else:
            input_pad = input
        setattr(self, 'input_pad', input_pad)
        out_h = (height - self.ksize) // self.stride + 1
        out_w = (width - self.ksize) // self.stride + 1
        output = np.zeros(shape=(input.shape[0], input.shape[1], out_h, out_w), dtype=input.dtype)
        for ih in range(out_h):
            for iw in range(out_w):
                output[:, :, ih, iw] = np.mean(
                    input_pad[
                        :, :,
                        ih * self.stride:ih * self.stride + self.ksize,
                        iw * self.stride:iw * self.stride + self.ksize
                    ], axis=(2, 3)
                )
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert hasattr(self, 'input_pad')
        out_grad = np.zeros_like(self.input_pad, dtype=np.float64)
        for ih in range(grad.shape[2]):
            for iw in range(grad.shape[3]):
                out_grad[
                    :, :,
                    ih * self.stride: ih * self.stride + self.ksize,
                    iw * self.stride: iw * self.stride + self.ksize
                ] += (grad[:, :, ih, iw] / self.ksize ** 2)[:, :, np.newaxis, np.newaxis]
        if self.padding == 0:
            return out_grad
        return out_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]


class MaxPool2d(Module):
    name = 'MaxPool2d'

    def __init__(self, ksize: int, stride: int, padding: int):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.padding = padding

    def forward(self, input: np.ndarray) -> np.ndarray:
        height = input.shape[2] + 2 * self.padding
        width = input.shape[3] + 2 * self.padding
        if self.padding > 0:
            input_pad = np.zeros((input.shape[0], input.shape[1], height, width), dtype=input.dtype)
            input_pad[:, :, self.padding:-self.padding, self.padding:-self.padding] = input
        else:
            input_pad = input
        setattr(self, 'input_pad', input_pad)
        out_h = (height - self.ksize) // self.stride + 1
        out_w = (width - self.ksize) // self.stride + 1
        output = np.zeros(shape=(input.shape[0], input.shape[1], out_h, out_w), dtype=input.dtype)
        for ih in range(out_h):
            for iw in range(out_w):
                output[:, :, ih, iw] = np.max(
                    input_pad[
                        :, :,
                        ih * self.stride:ih * self.stride + self.ksize,
                        iw * self.stride:iw * self.stride + self.ksize
                    ], axis=(2, 3)
                )
        setattr(self, 'output', output)
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert hasattr(self, 'input_pad')
        out_grad = np.zeros_like(self.input_pad, dtype=np.float64)
        for ih in range(grad.shape[2]):
            for iw in range(grad.shape[3]):
                ihl, ihr = ih * self.stride, ih * self.stride + self.ksize,
                iwl, iwr = iw * self.stride, iw * self.stride + self.ksize
                out_grad[
                    :, :, ihl:ihr, iwl:iwr
                ] += (grad[:, :, ih, iw][:, :, np.newaxis, np.newaxis]
                    * (self.input_pad[:, :, ihl:ihr, iwl:iwr] == self.output[:, :, ih, iw][:, :, np.newaxis, np.newaxis])
                )
        if self.padding == 0:
            return out_grad
        return out_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]


class GlobalAvgPool2d(Module):
    name = 'GlobalAvgPool2d'

    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        setattr(self, 'input', input)
        return input.mean(axis=(2, 3))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert hasattr(self, 'input')
        out_grad = np.zeros_like(self.input)
        out_grad[:, :, :, :] = grad[:, :, np.newaxis, np.newaxis] / (self.input.shape[2] * self.input.shape[3])
        return out_grad


class ReLU(Module):
    name = 'ReLU'

    def __init__(self):
        super().__init__()
        ...

    def forward(self, input: np.ndarray) -> np.ndarray:
        setattr(self, 'input', input)
        return np.maximum(input, 0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert hasattr(self, 'input')
        return grad * (self.input > 0)


class Sequential(Module):
    name = 'Sequential'

    def __init__(self, *args: Module):
        super().__init__()
        self._modules.extend(args)

    def forward(self, input: np.ndarray) -> np.ndarray:
        for module in self._modules:
            input = module.forward(input)
        return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for module in self._modules[::-1]:
            grad = module.backward(grad)
        return grad

    def train(self):
        for module in self._modules:
            module.train()

    def eval(self):
        for module in self._modules:
            module.eval()

class Flatten(Module):
    name = 'Flatten'

    def __init__(self):
        super().__init__()

    def forward(self, input: np.ndarray) -> np.ndarray:
        setattr(self, 'input_shape', input.shape)
        return input.reshape(input.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert hasattr(self, 'input_shape')
        return grad.reshape(self.input_shape)


class Dropout(Module):
    name = 'Dropout'

    def __init__(self, p: float):
        """
        :param p: 1 - 丢失的概率
        """
        super().__init__()
        self.do_dropout = True
        self.p = p

    def forward(self, input: np.ndarray) -> np.ndarray:
        if not self.do_dropout:
            return input
        mask = np.random.rand(*input.shape) < self.p
        setattr(self, 'mask', mask)
        return input * mask / self.p

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if not self.do_dropout:
            return grad
        assert hasattr(self, 'mask')
        return grad * self.mask * self.p

    def train(self):
        self.do_dropout = True

    def eval(self):
        self.do_dropout = False


if __name__ == '__main__':
    ...
