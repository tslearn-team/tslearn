"""The Numpy backend."""

import numpy as _np

from .base_backend import BaseBackend


class NumPyBackend(BaseBackend):
    """Class for the Numpy  backend."""

    def __init__(self):
        self.linalg = NumPyLinalg()
        self.random = NumPyRandom()

        self.int8 = _np.int8
        self.int16 = _np.int16
        self.int32 = _np.int32
        self.int64 = _np.int64
        self.float32 = _np.float32
        self.float64 = _np.float64
        self.complex64 = _np.complex64
        self.complex128 = _np.complex128

        self.all = _np.all
        self.array = _np.array
        self.dbl_max = _np.finfo("double").max
        self.diag = _np.diag
        self.empty = _np.empty
        self.exp = _np.exp
        self.hstack = _np.hstack
        self.inf = _np.inf
        self.iscomplex = _np.iscomplex
        self.isnan = _np.isnan
        self.log = _np.log
        self.max = _np.max
        self.mean = _np.mean
        self.median = _np.median
        self.min = _np.min
        self.nan = _np.nan
        self.reshape = _np.reshape
        self.shape = _np.shape
        self.sqrt = _np.sqrt
        self.vstack = _np.vstack
        self.zeros = _np.zeros
        self.zeros_like = _np.zeros_like

    @staticmethod
    def is_array(x):
        return type(x) is _np.ndarray

    @staticmethod
    def is_float(x):
        return isinstance(x, (_np.floating, float))

    @staticmethod
    def is_float32(x):
        return isinstance(x, _np.float32)

    @staticmethod
    def is_float64(x):
        return isinstance(x, _np.float64)

    @staticmethod
    def ndim(x):
        return x.ndim

    @staticmethod
    def to_float(x):
        return x.astype(float)

    @staticmethod
    def to_float32(x):
        return x.astype(_np.float32)

    @staticmethod
    def to_float64(x):
        return x.astype(_np.float64)

    @staticmethod
    def to_numpy(x):
        return x


class NumPyLinalg:
    def __init__(self):
        self.inv = _np.linalg.inv


class NumPyRandom:
    def __init__(self):
        self.rand = _np.random.rand
        self.randint = _np.random.randint
        self.randn = _np.random.randn
