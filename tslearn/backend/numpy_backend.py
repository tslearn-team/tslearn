"""The Numpy backend."""

import numpy as _np

from .base_backend import BaseBackend


class NumpyBackend(BaseBackend):
    """Class for the Numpy  backend."""

    def __init__(self):
        self.linalg = NumpyLinalg()

    @staticmethod
    def array(data, dtype=None):
        return _np.array(data, dtype=dtype)

    @staticmethod
    def exp(data, dtype=None):
        return _np.exp(data)

    @staticmethod
    def log(data, dtype=None):
        return _np.log(data)

    @staticmethod
    def shape(data):
        return _np.shape(data)

    @staticmethod
    def to_numpy(x):
        return x

    @staticmethod
    def zeros(shape, dtype=None):
        return _np.zeros(shape, dtype=dtype)


class NumpyLinalg:
    @staticmethod
    def inv(x):
        return _np.linalg.inv(x)
