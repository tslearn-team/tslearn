"""The generic backend."""

from tslearn.backend.numpy_backend import NumPyBackend
from tslearn.backend.pytorch_backend import PyTorchBackend


def select_backend(data):
    """Select backend.

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
        return NumPyBackend()
    elif "torch" in f"{type(data)}" or f"{data}".lower() == "pytorch":
        return PyTorchBackend()
    else:
        raise NotImplementedError("Not implemented backend")


def backend_to_string(backend):
    if "NumPY" in f"{backend}":
        return "numpy"
    return "pytorch"


class Backend(object):
    """Class for the  backend.

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
        self.backend_string = backend_to_string(self.backend)
        self.is_numpy = self.backend_string == "numpy"
        self.is_pytorch = self.backend_string == "pytorch"
        self.linalg = self.backend.linalg
        self.dbl_max = self.backend.dbl_max

    def array(self, data, dtype=None):
        return self.backend.array(data, dtype=dtype)

    def exp(self, data, dtype=None):
        return self.backend.exp(data, dtype=dtype)

    def get_backend(self):
        return self.backend

    def log(self, data, dtype=None):
        return self.backend.log(data, dtype=dtype)

    def set_backend(self, data=None):
        self.backend = select_backend(data)

    def shape(self, data):
        return self.backend.shape(data)

    def to_numpy(self, data):
        return self.backend.to_numpy(data)

    def zeros(self, shape, dtype=None):
        return self.backend.zeros(shape, dtype=dtype)
