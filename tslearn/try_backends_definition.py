import numpy as _np
import torch as _torch


class Backend(object):
    def __init__(self):
        pass

    @staticmethod
    def array(data, dtype=None):
        return NotImplementedError("Not implemented")

    @staticmethod
    def exp(data, dtype=None):
        return NotImplementedError("Not implemented")

    @staticmethod
    def log(data, dtype=None):
        return NotImplementedError("Not implemented")


class NumpyBackend(Backend):
    @staticmethod
    def array(data, dtype=None):
        return _np.array(data, dtype=dtype)

    @staticmethod
    def exp(data):
        return _np.exp(data)

    @staticmethod
    def log(data):
        return _np.log(data)


class PyTorchBackend(Backend):
    @staticmethod
    def array(data, dtype=_torch.float32):
        return _torch.tensor(data, dtype=dtype)

    @staticmethod
    def exp(data):
        return _torch.exp(data)

    @staticmethod
    def log(data):
        return _torch.log(data)


class GenericBackend(object):
    def __init__(self):
        self.backend = Backend()

    def get_backend(self, data):
        if "numpy" in f"{type(data)}":
            self.backend = NumpyBackend()
            return NumpyBackend()
        if "torch" in f"{type(data)}":
            self.backend = PyTorchBackend()
            return PyTorchBackend()

    def array(self, data, dtype=None):
        return self.backend.array(data, dtype=dtype)

    def exp(self, data):
        return self.backend.exp(data)

    def log(self, data):
        return self.backend.log(data)
