"""Classes for the backends."""

import numpy as _np
import torch as _torch


class Backend(object):
    def __init__(self):
        self.linalg = BackendLinalg()

    @staticmethod
    def shape(data):
        return NotImplementedError("Not implemented")

    @staticmethod
    def array(data, dtype=None):
        return NotImplementedError("Not implemented")

    @staticmethod
    def exp(data, dtype=None):
        return NotImplementedError("Not implemented")

    @staticmethod
    def log(data, dtype=None):
        return NotImplementedError("Not implemented")

    @staticmethod
    def zeros(shape, dtype=None):
        return NotImplementedError("Not implemented")


class BackendLinalg:
    @staticmethod
    def inv(x):
        return NotImplementedError("Not implemented")


class NumpyBackend(Backend):
    def __init__(self):
        self.linalg = NumpyLinalg()

    @staticmethod
    def shape(data):
        return _np.shape(data)

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
    def zeros(shape, dtype=None):
        return _np.zeros(shape, dtype=dtype)


class NumpyLinalg:
    @staticmethod
    def inv(x):
        return _np.linalg.inv(x)


class PyTorchBackend(Backend):
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


def select_backend(data):
    """Class for the generic backend.

    Parameter
    ---------
    data : array-like or string or None
        Indicates the backend to choose.
        Optional, default equals None.

    Returns
    -------
    backend : class
        The backend class.
        If data is a Numpy array or data equals 'numpy' or data is None,
        backend equals NumpyBackend().
        If data is a PyTorch array or data equals 'pytorch',
        backend equals PytorchBackend().
    """
    if "numpy" in f"{type(data)}" or f"{data}".lower() == "numpy" or data is None:
        return NumpyBackend()
    elif "torch" in f"{type(data)}" or f"{data}".lower() == "pytorch":
        return PyTorchBackend()
    else:
        raise NotImplementedError("Not implemented backend")


class GenericBackend(object):
    """Class for the generic backend.

    Parameter
    ---------
    data : array-like or string or None
        Indicates the backend to choose.
        If data is a Numpy array or data equals 'numpy' or data is None,
        self.backend is set to NumpyBackend().
        If data is a PyTorch array or data equals 'pytorch',
        self.backend is set to PytorchBackend().
        Optional, default equals None.
    """

    def __init__(self, data=None):
        self.backend = select_backend(data)
        self.linalg = self.backend.linalg

    def set_backend(self, data=None):
        self.backend = select_backend(data)

    def get_backend(self):
        return self.backend

    def shape(self, data):
        return self.backend.shape(data)

    def array(self, data, dtype=None):
        return self.backend.array(data, dtype=dtype)

    def exp(self, data, dtype=None):
        return self.backend.exp(data, dtype=dtype)

    def log(self, data, dtype=None):
        return self.backend.log(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return self.backend.zeros(shape, dtype=dtype)
