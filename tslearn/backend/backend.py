"""The generic backend."""

from tslearn.backend.numpy_backend import NumPyBackend

try:
    import torch

    from tslearn.backend.pytorch_backend import PyTorchBackend
except ImportError:

    class PyTorchBackend:
        def __init__(self):
            raise ValueError(
                "Could not use pytorch backend since torch is not installed"
            )


def instantiate_backend(*args):
    """Select backend.

    Parameter
    ---------
    *args : Input arguments can be Backend instance or string or array or None
        Arguments used to define the backend instance.

    Returns
    -------
    backend : Backend instance
        The backend instance.
    """
    for arg in args:
        if isinstance(arg, Backend):
            return arg
        if "numpy" in f"{type(arg)}" or "numpy" in f"{arg}".lower():
            return Backend("numpy")
        if "torch" in f"{type(arg)}" or "torch" in f"{arg}".lower():
            return Backend("pytorch")
    return Backend("numpy")


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
    if "torch" in f"{type(data)}" or "torch" in f"{data}".lower():
        return PyTorchBackend()
    return NumPyBackend()


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
        self.backend_string = self.backend.backend_string
        self.is_numpy = self.backend_string == "numpy"
        self.is_pytorch = self.backend_string == "pytorch"

        self.linalg = self.backend.linalg
        self.random = self.backend.random
        self.testing = self.backend.testing

        self.int8 = self.backend.int8
        self.int16 = self.backend.int16
        self.int32 = self.backend.int32
        self.int64 = self.backend.int64
        self.float32 = self.backend.float32
        self.float64 = self.backend.float64
        self.complex64 = self.backend.complex64
        self.complex128 = self.backend.complex128

        self.abs = self.backend.abs
        self.all = self.backend.all
        self.any = self.backend.any
        self.arange = self.backend.arange
        self.argmax = self.backend.argmax
        self.argmin = self.backend.argmin
        self.array = self.backend.array
        self.cast = self.backend.cast
        self.cdist = self.backend.cdist
        self.ceil = self.backend.ceil
        self.copy = self.backend.copy
        self.dbl_max = self.backend.dbl_max
        self.diag = self.backend.diag
        self.empty = self.backend.empty
        self.exp = self.backend.exp
        self.eye = self.backend.eye
        self.floor = self.backend.floor
        self.from_numpy = self.backend.from_numpy
        self.full = self.backend.full
        self.hstack = self.backend.hstack
        self.inf = self.backend.inf
        self.is_array = self.backend.is_array
        self.iscomplex = self.backend.iscomplex
        self.isfinite = self.backend.isfinite
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
        self.pairwise_distances = self.backend.pairwise_distances
        self.pairwise_euclidean_distances = self.backend.pairwise_euclidean_distances
        self.pdist = self.backend.pdist
        self.reshape = self.backend.reshape
        self.round = self.backend.round
        self.shape = self.backend.shape
        self.sqrt = self.backend.sqrt
        self.sum = self.backend.sum
        self.to_numpy = self.backend.to_numpy
        self.tril = self.backend.tril
        self.tril_indices = self.backend.tril_indices
        self.triu = self.backend.triu
        self.triu_indices = self.backend.triu_indices
        self.vstack = self.backend.vstack
        self.zeros = self.backend.zeros
        self.zeros_like = self.backend.zeros_like

    def get_backend(self):
        return self.backend

    def set_backend(self, data=None):
        self.backend = select_backend(data)
