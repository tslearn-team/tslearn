"""The JAX backend."""

import numpy as _np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

try:
    import jax.numpy as _jnp

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

            self.abs = _jnp.abs
            self.all = _jnp.all
            self.any = _jnp.any
            self.arange = _jnp.arange
            self.argmax = _jnp.argmax
            self.argmin = _jnp.argmin
            self.array = _jnp.array
            self.cdist = cdist
            self.ceil = _jnp.ceil
            self.dbl_max = _jnp.finfo("double").max
            self.diag = _jnp.diag
            self.empty = _jnp.empty
            self.exp = _jnp.exp
            self.eye = _jnp.eye
            self.floor = _jnp.floor
            self.full = _jnp.full
            self.hstack = _jnp.hstack
            self.inf = _jnp.inf
            self.iscomplex = _jnp.iscomplex
            self.isfinite = _jnp.isfinite
            self.isnan = _jnp.isnan
            self.log = _jnp.log
            self.max = _jnp.max
            self.mean = _jnp.mean
            self.median = _jnp.median
            self.min = _jnp.min
            self.nan = _jnp.nan
            self.pairwise_distances = pairwise_distances
            self.pairwise_euclidean_distances = euclidean_distances
            self.pdist = pdist
            self.reshape = _jnp.reshape
            self.round = _jnp.round
            self.shape = _jnp.shape
            self.sqrt = _jnp.sqrt
            self.sum = _jnp.sum
            self.tril = _jnp.tril
            self.tril_indices = _jnp.tril_indices
            self.triu = _jnp.triu
            self.triu_indices = _jnp.triu_indices
            self.vstack = _jnp.vstack
            self.zeros = _jnp.zeros
            self.zeros_like = _jnp.zeros_like

        @staticmethod
        def belongs_to_backend(x):
            return "jax" in str(type(x)).lower()

        @staticmethod
        def cast(x, dtype):
            return x.astype(dtype)

        @staticmethod
        def copy(x):
            return x.copy()

        @staticmethod
        def from_numpy(x):
            return x

        @staticmethod
        def is_array(x):
            return type(x) is _jnp.ndarray

        @staticmethod
        def is_float(x):
            return isinstance(x, (_jnp.floating, float))

        @staticmethod
        def is_float32(x):
            return isinstance(x, _jnp.float32)

        @staticmethod
        def is_float64(x):
            return isinstance(x, _jnp.float64)

        @staticmethod
        def ndim(x):
            return x.ndim

        @staticmethod
        def to_numpy(x):
            return x


    class JAXLinalg:
        def __init__(self):
            self.inv = _jnp.linalg.inv
            self.norm = _jnp.linalg.norm


    class JAXRandom:
        def __init__(self):
            self.rand = _jnp.random.rand
            self.randint = _jnp.random.randint
            self.randn = _jnp.random.randn


    class JAXTesting:
        def __init__(self):
            self.assert_allclose = _np.testing.assert_allclose
            self.assert_equal = _np.testing.assert_equal
