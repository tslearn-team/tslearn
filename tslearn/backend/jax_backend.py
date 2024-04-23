"""The JAX backend."""

import numpy as _np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

try:
    import jax as _jax
    import jax.numpy as _jnp
    from jax import config

    config.update("jax_enable_x64", True)

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

if not HAS_JAX:

    class JAXBackend:
        def __init__(self):
            raise ValueError(
                "Could not use the JAX backend since JAX is not installed."
            )

else:

    class JAXMutableArray:
        def __init__(self, *args, **kwargs):
            if len(args) + len(kwargs) == 0:
                self.array = None
            else:
                self.array = _jnp.array(*args, **kwargs)

        dtype = property(lambda self: self.array.dtype)
        ndim = property(lambda self: self.array.ndim)
        shape = property(lambda self: self.array.shape)
        T = property(lambda self: self.from_jnp_array(self.array.T))

        @classmethod
        def from_jnp_array(cls, arr):
            jni = cls()
            jni.array = arr
            return jni

        def __abs__(self):
            return self.from_jnp_array(abs(self.array))

        def __add__(self, other):
            return self.from_jnp_array(self.array + other)

        def __array__(self, *args, **kwargs):
            return _np.array(self.array, *args, **kwargs)

        def __bool__(self):
            return self.array.__bool__()

        def __div__(self, other):
            return self.from_jnp_array(self.array.__div__(other))

        def __eq__(self, other):
            return self.from_jnp_array(self.array == other)

        def __float__(self):
            return self.array.__float__()

        def __floordiv__(self, other):
            return self.from_jnp_array(self.array // other)

        def __ge__(self, other):
            return self.from_jnp_array(self.array >= other)

        def __getitem__(self, key):
            return self.from_jnp_array(self.array.at[key].get())

        def __gt__(self, other):
            return self.from_jnp_array(self.array > other)

        def __index__(self):
            return self.from_jnp_array(self.array.__index__())

        def __int__(self):
            return self.array.__int__()

        def __invert__(self):
            return self.from_jnp_array(self.array.__invert__())

        def __iter__(self):
            return self.from_jnp_array(self.array.__iter__())

        def __jax_array__(self):
            return self.array

        def __len__(self):
            return self.array.__len__()

        def __le__(self, other):
            return self.from_jnp_array(self.array <= other)

        def __lt__(self, other):
            return self.from_jnp_array(self.array < other)

        def __lshift__(self, other):
            return self.from_jnp_array(self.array << other)

        def __matmul__(self, other):
            return self.from_jnp_array(self.array @ other)

        def __mod__(self, other):
            return self.from_jnp_array(self.array % other)

        def __mul__(self, other):
            return self.from_jnp_array(self.array * other)

        def __ne__(self, other):
            return self.from_jnp_array(self.array != other)

        def __neg__(self):
            return self.from_jnp_array(self.array.__neg__())

        def __next__(self):
            return self.from_jnp_array(self.array.__next__())

        def __or__(self, other):
            return self.from_jnp_array(self.array | other)

        def __pos__(self):
            return self.from_jnp_array(self.array.__pos__())

        def __pow__(self, other):
            return self.from_jnp_array(self.array ** other)

        def __radd__(self, other):
            return self.from_jnp_array(other + self.array)

        def __rdiv__(self, other):
            return self.from_jnp_array(self.array.__rdiv__(other))

        def __repr__(self):
            return self.array.__repr__()

        def __rmul__(self, other):
            return self.from_jnp_array(other * self.array)

        def __rshift__(self, other):
            return self.from_jnp_array(self.array >> other)

        def __rsub__(self, other):
            return self.from_jnp_array(other - self.array)

        def __rtruediv__(self, other):
            return self.from_jnp_array(other / self.array)

        def __setitem__(self, key, value):
            if hasattr(key, 'array'):
                key = key.array
            self.array = self.array.at[key].set(value)

        def __sub__(self, other):
            return self.from_jnp_array(self.array - other)

        def __truediv__(self, other):
            return self.from_jnp_array(self.array.__truediv__(other))

        def __xor__(self, other):
            return self.from_jnp_array(self.array ^ other)

        def astype(self, dtype):
            self.array.astype(dtype)

        def conj(self):
            return self.from_jnp_array(self.array.conj())

        def copy(self):
            return self.from_jnp_array(self.array.copy())

        def reshape(self, shape, order='C'):
            return self.from_jnp_array(self.array.reshape(shape, order=order))
            # self.array.reshape(shape, order=order)

        def tolist(self):
            return self.array.tolist()


    class JAXBackend(object):
        """Class for the JAX  backend."""

        def __init__(self):
            self.backend_string = "jax"

            self.linalg = JAXLinalg()
            self.random = JAXRandom()
            self.testing = JAXTesting()

            self.int8 = _jnp.int8
            self.int16 = _jnp.int16
            self.int32 = _jnp.int32
            self.int64 = _jnp.int64
            self.float32 = _jnp.float32
            self.float64 = _jnp.float64
            self.complex64 = _jnp.complex64
            self.complex128 = _jnp.complex128

            self.dbl_max = _jnp.finfo("double").max
            self.inf = _jnp.inf
            self.nan = _jnp.nan

        @staticmethod
        def abs(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.abs(*args, **kwargs))

        @staticmethod
        def all(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.all(*args, **kwargs))

        @staticmethod
        def any(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.any(*args, **kwargs))

        @staticmethod
        def arange(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.arange(*args, **kwargs))

        @staticmethod
        def argmax(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.argmax(*args, **kwargs))

        @staticmethod
        def argmin(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.argmin(*args, **kwargs))

        @staticmethod
        def array(*args, **kwargs):
            return JAXMutableArray(*args, **kwargs)

        @staticmethod
        def belongs_to_backend(x):
            return "jax" in str(type(x)).lower()

        def cast(self, x, dtype):
            return self.array(x, dtype=dtype)

        @staticmethod
        def ceil(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.ceil(*args, **kwargs))

        @staticmethod
        def cos(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.cos(*args, **kwargs))

        @staticmethod
        def copy(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.copy(*args, **kwargs))

        def cdist(self, XA, XB, metric='euclidean', p=2):
            if metric == "euclidean":
                metric = lambda x, y: self.linalg.norm(x - y)
            if metric == "minkowski":
                metric = lambda x, y: self.linalg.norm(x - y, ord=p) ** 2
            if metric == "sqeuclidean":
                metric = lambda x, y: self.linalg.norm(x - y) ** 2
            if metric == "chebyshev":
                metric = lambda x, y: self.linalg.norm(x - y, ord=self.inf) ** 2
            if callable(metric):
                distance_matrix = self.zeros((XA.shape[0], XB.shape[0]))
                for i in range(XA.shape[0]):
                    for j in range(XB.shape[0]):
                        distance_matrix[i, j] = metric(XA[i, ...], XB[j, ...])
                return distance_matrix
            raise ValueError(f"Metric {metric} not implemented in JAX backend.")

        @staticmethod
        def diag(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.diag(*args, **kwargs))

        @staticmethod
        def empty(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.empty(*args, **kwargs))

        @staticmethod
        def exp(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.exp(*args, **kwargs))

        @staticmethod
        def eye(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.eye(*args, **kwargs))

        @staticmethod
        def floor(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.floor(*args, **kwargs))

        @staticmethod
        def from_numpy(x):
            return JAXMutableArray.from_jnp_array(_jnp.array(x))

        @staticmethod
        def full(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.full(*args, **kwargs))

        @staticmethod
        def full_like(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.full_like(*args, **kwargs))

        @staticmethod
        def hstack(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.hstack(*args, **kwargs))

        @staticmethod
        def is_array(x):
            return type(x) is JAXMutableArray

        @staticmethod
        def isclose(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.isclose(*args, **kwargs))

        @staticmethod
        def iscomplex(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.iscomplex(*args, **kwargs))

        @staticmethod
        def isfinite(x):
            return JAXMutableArray.from_jnp_array(_jnp.isfinite(x))

        @staticmethod
        def is_float(x):
            if hasattr(x, 'dtype'):
                return 'float' in str(x.dtype)
            return 'float' in str(x) + str(type(x))

        @staticmethod
        def is_float32(x):
            if hasattr(x, 'dtype'):
                return 'float32' in str(x.dtype)
            return 'float32' in str(x) + str(type(x))

        @staticmethod
        def is_float64(x):
            if hasattr(x, 'dtype'):
                return 'float64' in str(x.dtype)
            return 'float64' in str(x) + str(type(x))

        @staticmethod
        def isnan(x):
            return JAXMutableArray.from_jnp_array(_jnp.isnan(x))

        @staticmethod
        def log(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.log(*args, **kwargs))

        @staticmethod
        def mean(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.mean(*args, **kwargs))

        @staticmethod
        def median(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.median(*args, **kwargs))

        @staticmethod
        def max(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.max(*args, **kwargs))

        @staticmethod
        def min(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.min(*args, **kwargs))

        @staticmethod
        def ndim(a):
            return _jnp.ndim(a)

        @staticmethod
        def ones(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.ones(*args, **kwargs))

        def pairwise_distances(self, X, Y=None, metric="euclidean"):
            if Y is None:
                Y = X
            if metric == "euclidean":
                metric = lambda x, y: self.linalg.norm(x - y)
            if metric == "sqeuclidean":
                metric = lambda x, y: self.linalg.norm(x - y) ** 2
            if callable(metric):
                distance_matrix = self.zeros((X.shape[0], Y.shape[0]))
                for i in range(X.shape[0]):
                    for j in range(Y.shape[0]):
                        distance_matrix[i, j] = metric(X[i, ...], Y[j, ...])
                return distance_matrix
            raise ValueError(f"Metric {metric} not implemented in JAX backend.")

        def pairwise_euclidean_distances(self, X, Y=None):
            return self.pairwise_distances(X=X, Y=Y, metric="euclidean")

        def pdist(self, x, metric="euclidean", p=2):
            if metric == "euclidean":
                metric = lambda x, y: self.linalg.norm(x - y)
            if metric == "minkowski":
                metric = lambda x, y: self.linalg.norm(x - y, ord=p) ** 2
            if metric == "sqeuclidean":
                metric = lambda x, y: self.linalg.norm(x - y) ** 2
            if metric == "chebyshev":
                metric = lambda x, y: self.linalg.norm(x - y, ord=self.inf) ** 2
            if callable(metric):
                n = x.shape[0]
                distances = self.zeros((n * (n - 1)) // 2)
                for i in range(n):
                    for j in range(i + 1, n):
                        distances[n * i + j - ((i + 2) * (i + 1)) // 2] = metric(x[i, ...], x[j, ...])
                return distances
            raise ValueError(f"Metric {metric} not implemented in JAX backend.")

        @staticmethod
        def reshape(a, newshape, order='C'):
            return _jnp.reshape(a, newshape, order)

        @staticmethod
        def round(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.round(*args, **kwargs))

        @staticmethod
        def shape(a):
            return _jnp.shape(a)

        @staticmethod
        def sin(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.sin(*args, **kwargs))

        @staticmethod
        def sqrt(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.sqrt(*args, **kwargs))

        @staticmethod
        def sum(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.sum(*args, **kwargs))

        @staticmethod
        def to_numpy(x):
            return _np.array(_jnp.array(x))

        @staticmethod
        def transpose(a, axes=None):
            return JAXMutableArray.from_jnp_array(_jnp.transpose(a=a, axes=axes))

        @staticmethod
        def tril(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.tril(*args, **kwargs))

        @staticmethod
        def tril_indices(n, k=0, m=None):
            return _jnp.tril_indices(n=n, k=k, m=m)

        @staticmethod
        def triu(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.triu(*args, **kwargs))

        @staticmethod
        def triu_indices(n, k=0, m=None):
            return _jnp.triu_indices(n=n, k=k, m=m)

        @staticmethod
        def vstack(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.vstack(*args, **kwargs))

        @staticmethod
        def zeros(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.zeros(*args, **kwargs))

        @staticmethod
        def zeros_like(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.zeros_like(*args, **kwargs))


    class JAXLinalg:

        @staticmethod
        def inv(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.linalg.inv(*args, **kwargs))

        @staticmethod
        def norm(*args, **kwargs):
            return JAXMutableArray.from_jnp_array(_jnp.linalg.norm(*args, **kwargs))

    class JAXRandom:
        def __init__(self):
            self.key = _jax.random.PRNGKey(0)

        def rand(self, *args, dtype=float, key=None):
            if key is None:
                self.key = _jax.random.split(self.key, num=1)[0, :]
                key = self.key
            return JAXMutableArray.from_jnp_array(_jax.random.uniform(
                key=key, shape=args, dtype=dtype, minval=0.0, maxval=1.0
            ))

        def randint(self, low, high=None, size=(), dtype=int, key=None):
            if key is None:
                self.key = _jax.random.split(self.key, num=1)[0, :]
                key = self.key
            if high is None:
                minval = 0
                maxval = low
            else:
                minval = low
                maxval = high
            return JAXMutableArray.from_jnp_array(_jax.random.randint(
                key=key, shape=size, minval=minval, maxval=maxval, dtype=dtype
            ))

        def randn(self, *args, dtype=float, key=None):
            if key is None:
                self.key = _jax.random.split(self.key, num=1)[0, :]
                key = self.key
            return JAXMutableArray.from_jnp_array(_jax.random.normal(
                key=key, shape=args, dtype=dtype
            ))

        def uniform(self, low=0.0, high=1.0, size=(), dtype=float, key=None):
            if key is None:
                self.key = _jax.random.split(self.key, num=1)[0, :]
                key = self.key
            return JAXMutableArray.from_jnp_array(_jax.random.uniform(
                key=key, shape=size, dtype=dtype, minval=low, maxval=high
            ))

    class JAXTesting:
        def __init__(self):
            self.assert_equal = _np.testing.assert_equal

        @staticmethod
        def assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True, err_msg='', verbose=True):
            return _np.testing.assert_allclose(
                actual=JAXBackend.to_numpy(actual),
                desired=JAXBackend.to_numpy(desired),
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                err_msg=err_msg,
                verbose=verbose)
