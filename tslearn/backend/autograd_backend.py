"""The Autograd backend.

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
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances

try:
    import autograd.numpy as _adnp

    HAS_AUTOGRAD = True
except ImportError:
    HAS_AUTOGRAD = False


if not HAS_AUTOGRAD:

    class AutogradBackend:
        def __init__(self):
            raise ValueError(
                "Could not use the Autograd backend since Autograd is not installed."
            )

else:

    class AutogradBackend(object):
        """Class for the Autograd  backend."""

        def __init__(self):
            self.backend_string = "autograd"

            self.linalg = AutogradLinalg()
            self.random = AutogradRandom()
            self.testing = AutogradTesting()

            self.int8 = _adnp.int8
            self.int16 = _adnp.int16
            self.int32 = _adnp.int32
            self.int64 = _adnp.int64
            self.float32 = _adnp.float32
            self.float64 = _adnp.float64
            self.complex64 = _adnp.complex64
            self.complex128 = _adnp.complex128

            self.abs = _adnp.abs
            self.all = _adnp.all
            self.any = _adnp.any
            self.arange = _adnp.arange
            self.argmax = _adnp.argmax
            self.argmin = _adnp.argmin
            self.array = _adnp.array
            self.cdist = cdist
            self.ceil = _adnp.ceil
            self.dbl_max = _adnp.finfo("double").max
            self.diag = _adnp.diag
            self.empty = _adnp.empty
            self.exp = _adnp.exp
            self.eye = _adnp.eye
            self.floor = _adnp.floor
            self.full = _adnp.full
            self.hstack = _adnp.hstack
            self.inf = _adnp.inf
            self.iscomplex = _adnp.iscomplex
            self.isfinite = _adnp.isfinite
            self.isnan = _adnp.isnan
            self.log = _adnp.log
            self.max = _adnp.max
            self.mean = _adnp.mean
            self.median = _adnp.median
            self.min = _adnp.min
            self.nan = _adnp.nan
            self.pairwise_distances = pairwise_distances
            self.pairwise_euclidean_distances = euclidean_distances
            self.pdist = pdist
            self.reshape = _adnp.reshape
            self.round = _adnp.round
            self.shape = _adnp.shape
            self.sqrt = _adnp.sqrt
            self.sum = _adnp.sum
            self.tril = _adnp.tril
            self.tril_indices = _adnp.tril_indices
            self.triu = _adnp.triu
            self.triu_indices = _adnp.triu_indices
            self.vstack = _adnp.vstack
            self.zeros = _adnp.zeros
            self.zeros_like = _adnp.zeros_like

        @staticmethod
        def belongs_to_backend(x):
            return "autograd" in f"{type(x)}".lower()

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
            return type(x) is _adnp.ndarray

        @staticmethod
        def is_float(x):
            return isinstance(x, (_adnp.floating, float))

        @staticmethod
        def is_float32(x):
            return isinstance(x, _adnp.float32)

        @staticmethod
        def is_float64(x):
            return isinstance(x, _adnp.float64)

        @staticmethod
        def ndim(x):
            return x.ndim

        @staticmethod
        def to_numpy(x):
            return x


    class AutogradLinalg:
        def __init__(self):
            self.inv = _adnp.linalg.inv
            self.norm = _adnp.linalg.norm


    class AutogradRandom:
        def __init__(self):
            self.rand = _adnp.random.rand
            self.randint = _adnp.random.randint
            self.randn = _adnp.random.randn


    class AutogradTesting:
        def __init__(self):
            self.assert_allclose = _np.testing.assert_allclose
            self.assert_equal = _np.testing.assert_equal
