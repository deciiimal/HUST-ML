import numpy as np
from scipy.special import logsumexp, softmax

class CrossEntropyLoss:
    """
    LogSumExp
    """
    def __init__(self):
        ...

    def __call__(self, y_true, y_pred) -> np.ndarray:
        """
        :param y_true: 真实的分类
        :param y_pred: 未处理的预测
        """
        setattr(self, 'y_true', y_true)
        setattr(self, 'y_pred', y_pred)
        return - y_pred[np.arange(len(y_pred)), y_true] + logsumexp(y_pred, axis=1)

    @property
    def grad(self) -> np.ndarray:
        assert hasattr(self, 'y_true')
        hat = softmax(self.y_pred, axis=1)
        hat[np.arange(len(self.y_pred)), self.y_true] -= 1
        return hat
