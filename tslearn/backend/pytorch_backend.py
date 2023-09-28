"""The PyTorch backend.

Several backend functions are inspired from a python package about Machine Learning
in Riemannian manifolds named geomstats [JMLR:v21:19-027], also implementing several backends.

References
----------------
[JMLR:v21:19-027] Nina Miolane, Nicolas Guigui, Alice Le Brigant, Johan Mathe,
Benjamin Hou, Yann Thanwerdas, Stefan Heyder, Olivier Peltre, Niklas Koep, Hadi Zaatiti,
Hatem Hajri, Yann Cabanes, Thomas Gerald, Paul Chauchat, Christian Shewmake, Daniel Brooks,
Bernhard Kainz, Claire Donnat, Susan Holmes and Xavier Pennec.
Geomstats:  A Python Package for Riemannian Geometry in Machine Learning,
Journal of Machine Learning Research, 2020, volume 21, number 223, pages 1-9,
http://jmlr.org/papers/v21/19-027.html, https://github.com/geomstats/geomstats/
"""

import numpy as _np

try:
    import torch as _torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if not HAS_TORCH:

    class PyTorchBackend:
        def __init__(self):
            raise ValueError(
                "Could not use the PyTorch backend since torch is not installed"
            )

else:

    class PyTorchBackend(object):
        """Class for the PyTorch  backend."""

        def __init__(self):
            self.backend_string = "pytorch"

            self.linalg = PyTorchLinalg()
            self.random = PyTorchRandom()
            self.testing = PyTorchTesting()

            self.int8 = _torch.int8
            self.int16 = _torch.int16
            self.int32 = _torch.int32
            self.int64 = _torch.int64
            self.float32 = _torch.float32
            self.float64 = _torch.float64
            self.complex64 = _torch.complex64
            self.complex128 = _torch.complex128

            self.abs = _torch.abs
            self.any = _torch.any
            self.arange = _torch.arange
            self.argmax = _torch.argmax
            self.argmin = _torch.argmin
            self.dbl_max = _torch.finfo(_torch.double).max
            self.ceil = _torch.ceil
            self.diag = _torch.diag
            self.empty = _torch.empty
            self.exp = _torch.exp
            self.eye = _torch.eye
            self.floor = _torch.floor
            self.full = _torch.full
            self.hstack = _torch.hstack
            self.inf = _torch.inf
            self.is_array = _torch.is_tensor
            self.isfinite = _torch.isfinite
            self.isnan = _torch.isnan
            self.log = _torch.log
            self.max = _torch.max
            self.mean = _torch.mean
            self.median = _torch.median
            self.min = _torch.min
            self.nan = _torch.nan
            self.pairwise_euclidean_distances = _torch.cdist
            self.reshape = _torch.reshape
            self.round = _torch.round
            self.sum = _torch.sum
            self.vstack = _torch.vstack
            self.zeros = _torch.zeros
            self.zeros_like = _torch.zeros_like

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

        def array(self, val, dtype=None):
            if _torch.is_tensor(val):
                if dtype is None or val.dtype == dtype:
                    return val.clone()
                return self.cast(val, dtype=dtype)

            elif isinstance(val, _np.ndarray):
                tensor = self.from_numpy(val)
                if dtype is not None and tensor.dtype != dtype:
                    tensor = self.cast(tensor, dtype=dtype)

                return tensor

            elif isinstance(val, (list, tuple)) and len(val):
                tensors = [self.array(tensor, dtype=dtype) for tensor in val]
                return _torch.stack(tensors)

            return _torch.tensor(val, dtype=dtype)

        @staticmethod
        def belongs_to_backend(x):
            return "torch" in f"{type(x)}".lower()

        def cast(self, x, dtype):
            if _torch.is_tensor(x):
                return x.to(dtype=dtype)
            return self.array(x, dtype=dtype)

        @staticmethod
        def cdist(x, y, metric="euclidean", p=None):
            if metric == "euclidean":
                return _torch.cdist(x, y)
            if metric == "sqeuclidean":
                return _torch.cdist(x, y) ** 2
            if metric == "minkowski":
                return _torch.cdist(x, y, p=p)
            raise ValueError(f"Metric {metric} not implemented in PyTorch backend.")

        @staticmethod
        def copy(x):
            return x.clone()

        @staticmethod
        def from_numpy(x):
            return _torch.from_numpy(x)

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

        @staticmethod
        def pairwise_distances(X, Y=None, metric="euclidean"):
            if Y is None:
                Y = X
            if metric == "euclidean":
                return _torch.cdist(X, Y)
            if metric == "sqeuclidean":
                return _torch.cdist(X, Y) ** 2
            if callable(metric):
                distance_matrix = _torch.zeros(X.shape[0], Y.shape[0])
                for i in range(X.shape[0]):
                    for j in range(Y.shape[0]):
                        distance_matrix[i, j] = metric(X[i, ...], Y[j, ...])
                return distance_matrix
            raise ValueError(f"Metric {metric} not implemented in PyTorch backend.")

        @staticmethod
        def pdist(x, metric="euclidean", p=None):
            if metric == "euclidean":
                return _torch.pdist(x)
            if metric == "sqeuclidean":
                return _torch.pdist(x) ** 2
            if metric == "minkowski":
                return _torch.pdist(x, p=p)
            raise ValueError(f"Metric {metric} not implemented in PyTorch backend.")

        def shape(self, data):
            if not self.is_array(data):
                data = self.array(data)
            return tuple(_torch.Tensor.size(data))

        def sqrt(self, x, out=None):
            if not self.is_array(x):
                x = self.array(x)
            return _torch.sqrt(x, out=out)

        @staticmethod
        def to_numpy(x):
            return x.detach().cpu().numpy()

        @staticmethod
        def tril(mat, k=0):
            return _torch.tril(mat, diagonal=k)

        @staticmethod
        def tril_indices(n, k=0, m=None):
            if m is None:
                m = n
            x = _torch.tril_indices(row=n, col=m, offset=k)
            return x[0], x[1]

        @staticmethod
        def triu(mat, k=0):
            return _torch.triu(mat, diagonal=k)

        @staticmethod
        def triu_indices(n, k=0, m=None):
            if m is None:
                m = n
            x = _torch.triu_indices(row=n, col=m, offset=k)
            return x[0], x[1]

    class PyTorchLinalg:
        def __init__(self):
            self.inv = _torch.linalg.inv
            self.norm = _torch.linalg.norm

    class PyTorchRandom:
        def __init__(self):
            self.rand = _torch.rand
            self.randint = _torch.randint
            self.randn = _torch.randn

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

    class PyTorchTesting:
        def __init__(self):
            self.assert_allclose = _torch.allclose
            self.assert_equal = _torch.testing.assert_close
