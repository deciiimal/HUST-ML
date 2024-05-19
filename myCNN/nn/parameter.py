import numpy as np

class Parameter:
    def __init__(self, param: np.ndarray | None, name: str):
        self._name = name
        self.param = param
        if self.param is not None:
            self._grad = np.zeros_like(self.param)

    @property
    def grad(self):
        return self._grad

    @property
    def name(self) -> str:
        return self._name

    @grad.setter
    def grad(self, grad):
        self._grad = grad
        
    def zero_grad(self):
        self._grad[:] = 0.
        
    def __repr__(self):
        s = f'name: {self._name}, param: {self.param}'
        if hasattr(self, '_grad'):
            s += f', grad: {self._grad}'
        return s
        


