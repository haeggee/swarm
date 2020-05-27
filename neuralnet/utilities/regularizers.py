import numpy as np


class Regularizer(object):
    """Basic Regularizer class."""

    def __init__(self, reg=0, include_bias=True):
        super().__init__()
        self._lambda = reg
        self.include_bias = include_bias

    def set_lambda(self, reg):
        self._lambda = reg

    def get_lambda(self):
        return self._lambda

    def loss(self, w):
        return 0

    def gradient(self, w):
        return np.zeros_like(w)


class L2Regularizer(Regularizer):
    """L2Regularizer.

    L2 regularization is given by lambda @ w @ w.
    """

    def __init__(self, reg=0, include_bias=True):
        super().__init__(reg, include_bias)

    def loss(self, w):
        if self.include_bias:
            return self._lambda * np.square(np.linalg.norm(w[:-1], 2))
        return self._lambda * np.square(np.linalg.norm(w, 2))

    def gradient(self, w):
        if self.include_bias:
            gradient = np.zeros_like(w)
            gradient[:-1] = self._lambda * w[:-1]
        else:
            gradient = self._lambda * w 
        return gradient


class L1Regularizer(Regularizer):
    """L1Regularizer.

    L1 regularization is given by lambda @ |w|.
    """

    def __init__(self, reg=0, include_bias=True):
        super().__init__(reg, include_bias)

    def loss(self, w):
        if self.include_bias:
            return self._lambda * np.linalg.norm(w[:-1], 1)
        return self._lambda * np.linalg.norm(w, 1)

    def gradient(self, w):
        if self.include_bias:
            gradient = np.zeros_like(w)
            gradient[:-1] = self._lambda * np.sign(w[:-1])
        else:
            gradient = self._lambda * np.sign(w)
        return gradient
