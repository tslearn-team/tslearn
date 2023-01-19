"""The Numpy backend."""

import numpy as _np

from .base_backend import BaseBackend


class NumPyBackend(BaseBackend):
    """Class for the Numpy  backend."""

    def __init__(self):
        self.linalg = NumPyLinalg()
        self.random = NumPyRandom()
        self.testing = NumPyTesting()

        self.int8 = _np.int8
        self.int16 = _np.int16
        self.int32 = _np.int32
        self.int64 = _np.int64
        self.float32 = _np.float32
        self.float64 = _np.float64
        self.complex64 = _np.complex64
        self.complex128 = _np.complex128

        self.abs = _np.abs
        self.all = _np.all
        self.any = _np.any
        self.arange = _np.arange
        self.argmax = _np.argmax
        self.argmin = _np.argmin
        self.array = _np.array
        self.ceil = _np.ceil
        self.dbl_max = _np.finfo("double").max
        self.diag = _np.diag
        self.empty = _np.empty
        self.exp = _np.exp
        self.eye = _np.eye
        self.floor = _np.floor
        self.full = _np.full
        self.hstack = _np.hstack
        self.inf = _np.inf
        self.iscomplex = _np.iscomplex
        self.isfinite = _np.isfinite
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
        self.sum = _np.sum
        self.tril = _np.tril
        self.tril_indices = _np.tril_indices
        self.triu = _np.triu
        self.triu_indices = _np.triu_indices
        self.vstack = _np.vstack
        self.zeros = _np.zeros
        self.zeros_like = _np.zeros_like

    @staticmethod
    def cast(x, dtype):
        return x.astype(dtype)

    @staticmethod
    def from_numpy(x):
        return x

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


class NumPyTesting:
    def __init__(self):
        self.assert_allclose = _np.testing.assert_allclose
        self.assert_equal = _np.testing.assert_equal
