"""The PyTorch backend."""

import torch as _torch

from .base_backend import BaseBackend


class PyTorchBackend(BaseBackend):
    """Class for the PyTorch  backend."""

    def __init__(self):
        self.linalg = PyTorchLinalg()

    @staticmethod
    def shape(data):
        return tuple(_torch.Tensor.size(data))

    @staticmethod
    def array(data, dtype=None):
        return _torch.tensor(data, dtype=dtype)

    @staticmethod
    def exp(data, dtype=None):
        return _torch.exp(data)

    @staticmethod
    def log(data, dtype=None):
        return _torch.log(data)

    @staticmethod
    def zeros(shape, dtype=None):
        return _torch.zeros(shape, dtype=dtype)


class PyTorchLinalg:
    @staticmethod
    def inv(x):
        return _torch.linalg.inv(x)
