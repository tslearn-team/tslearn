"""The PyTorch backend."""

import torch as _torch

from .base_backend import BaseBackend


class PyTorchBackend(BaseBackend):
    """Class for the PyTorch  backend."""

    def __init__(self):
        self.linalg = PyTorchLinalg()
        self.random = PyTorchRandom()

        self.int8 = _torch.int8
        self.int16 = _torch.int16
        self.int32 = _torch.int32
        self.int64 = _torch.int64
        self.float32 = _torch.float32
        self.float64 = _torch.float64
        self.complex64 = _torch.complex64
        self.complex128 = _torch.complex128

        self.array = _torch.tensor
        self.dbl_max = _torch.finfo(_torch.double).max
        self.diag = _torch.diag
        self.empty = _torch.empty
        self.exp = _torch.exp
        self.hstack = _torch.hstack
        self.inf = _torch.inf
        self.is_array = _torch.is_tensor
        self.isnan = _torch.isnan
        self.log = _torch.log
        self.max = _torch.max
        self.mean = _torch.mean
        self.median = _torch.median
        self.min = _torch.min
        self.nan = _torch.nan
        self.sqrt = _torch.sqrt
        self.vstack = _torch.vstack
        self.zeros = _torch.zeros

    @staticmethod
    def all(x, axis=None):
        if not _torch.is_tensor(x):
            x = _torch.tensor(x)
        if axis is None:
            return x.bool().all()
        if isinstance(axis, int):
            return _torch.all(x.bool(), axis)
        if len(axis) == 1:
            return _torch.all(x, *axis)
        axis = list(axis)
        for i_axis, one_axis in enumerate(axis):
            if one_axis < 0:
                axis[i_axis] = x.ndim() + one_axis
        new_axis = tuple(k - 1 if k >= 0 else k for k in axis[1:])
        return all(_torch.all(x.bool(), axis[0]), new_axis)

    @staticmethod
    def iscomplex(x):
        if isinstance(x, complex):
            return True
        return x.dtype.is_complex

    @staticmethod
    def is_float(x):
        if isinstance(x, float):
            return True
        return x.dtype.is_floating_point

    @staticmethod
    def is_float32(x):
        return isinstance(x, _torch.float32)

    @staticmethod
    def is_float64(x):
        return isinstance(x, _torch.float64)

    @staticmethod
    def ndim(x):
        return x.dim()

    def shape(self, data):
        if not self.is_array(data):
            val = self.array(data)
        return tuple(_torch.Tensor.size(data))

    @staticmethod
    def to_float(data):
        return data.to(_torch.float)

    @staticmethod
    def to_float32(data):
        return data.to(_torch.float32)

    @staticmethod
    def to_float64(data):
        return data.to(_torch.float64)

    @staticmethod
    def to_numpy(x):
        return x.numpy()


class PyTorchLinalg:
    def __init__(self):
        self.inv = _torch.linalg.inv


class PyTorchRandom:
    def __init__(self):
        self.rand = _torch.rand
        self.randint = _torch.randint

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=(1,)):
        if not hasattr(size, "__iter__"):
            size = (size,)
        return _torch.normal(mean=loc, std=scale, size=size)

    @staticmethod
    def uniform(low=0.0, high=1.0, size=(1,), dtype=None):
        if not hasattr(size, "__iter__"):
            size = (size,)
        if low >= high:
            raise ValueError("Upper bound must be higher than lower bound")
        return (high - low) * _torch.rand(*size, dtype=dtype) + low
