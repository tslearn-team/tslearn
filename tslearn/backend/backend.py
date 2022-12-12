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
    if "NumPy" in f"{backend}":
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
        self.random = self.backend.random

        self.int8 = self.backend.int8
        self.int16 = self.backend.int16
        self.int32 = self.backend.int32
        self.int64 = self.backend.int64
        self.float32 = self.backend.float32
        self.float64 = self.backend.float64
        self.complex64 = self.backend.complex64
        self.complex128 = self.backend.complex128

        self.all = self.backend.all
        self.array = self.backend.array
        self.dbl_max = self.backend.dbl_max
        self.diag = self.backend.diag
        self.empty = self.backend.empty
        self.exp = self.backend.exp
        self.hstack = self.backend.hstack
        self.inf = self.backend.inf
        self.is_array = self.backend.is_array
        self.iscomplex = self.backend.iscomplex
        self.is_float = self.backend.is_float
        self.is_float32 = self.backend.is_float32
        self.is_float64 = self.backend.is_float64
        self.isnan = self.backend.isnan
        self.log = self.backend.log
        self.max = self.backend.max
        self.mean = self.backend.mean
        self.median = self.backend.median
        self.min = self.backend.min
        self.nan = self.backend.nan
        self.ndim = self.backend.ndim
        self.shape = self.backend.shape
        self.sqrt = self.backend.sqrt
        self.to_float = self.backend.to_float
        self.to_float32 = self.backend.to_float32
        self.to_float64 = self.backend.to_float64
        self.to_numpy = self.backend.to_numpy
        self.vstack = self.backend.vstack
        self.zeros = self.backend.zeros

    def get_backend(self):
        return self.backend

    def set_backend(self, data=None):
        self.backend = select_backend(data)
