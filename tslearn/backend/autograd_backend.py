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
    import autograd.numpy as _agnp

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

            self.int8 = _agnp.int8
            self.int16 = _agnp.int16
            self.int32 = _agnp.int32
            self.int64 = _agnp.int64
            self.float32 = _agnp.float32
            self.float64 = _agnp.float64
            self.complex64 = _agnp.complex64
            self.complex128 = _agnp.complex128

            self.abs = _agnp.abs
            self.all = _agnp.all
            self.any = _agnp.any
            self.arange = _agnp.arange
            self.argmax = _agnp.argmax
            self.argmin = _agnp.argmin
            self.array = _agnp.array
            self.cdist = cdist
            self.ceil = _agnp.ceil
            self.dbl_max = _agnp.finfo("double").max
            self.diag = _agnp.diag
            self.empty = _agnp.empty
            self.exp = _agnp.exp
            self.eye = _agnp.eye
            self.floor = _agnp.floor
            self.full = _agnp.full
            self.hstack = _agnp.hstack
            self.inf = _agnp.inf
            self.iscomplex = _agnp.iscomplex
            self.isfinite = _agnp.isfinite
            self.isnan = _agnp.isnan
            self.log = _agnp.log
            self.max = _agnp.max
            self.mean = _agnp.mean
            self.median = _agnp.median
            self.min = _agnp.min
            self.nan = _agnp.nan
            self.pairwise_distances = pairwise_distances
            self.pairwise_euclidean_distances = euclidean_distances
            self.pdist = pdist
            self.reshape = _agnp.reshape
            self.round = _agnp.round
            self.shape = _agnp.shape
            self.sqrt = _agnp.sqrt
            self.sum = _agnp.sum
            self.tril = _agnp.tril
            self.tril_indices = _agnp.tril_indices
            self.triu = _agnp.triu
            self.triu_indices = _agnp.triu_indices
            self.vstack = _agnp.vstack
            self.zeros = _agnp.zeros
            self.zeros_like = _agnp.zeros_like

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
            return type(x) is _agnp.ndarray

        @staticmethod
        def is_float(x):
            return isinstance(x, (_agnp.floating, float))

        @staticmethod
        def is_float32(x):
            return isinstance(x, _agnp.float32)

        @staticmethod
        def is_float64(x):
            return isinstance(x, _agnp.float64)

        @staticmethod
        def ndim(x):
            return x.ndim

        @staticmethod
        def to_numpy(x):
            return x


    class AutogradLinalg:
        def __init__(self):
            self.inv = _agnp.linalg.inv
            self.norm = _agnp.linalg.norm


    class AutogradRandom:
        def __init__(self):
            self.rand = _agnp.random.rand
            self.randint = _agnp.random.randint
            self.randn = _agnp.random.randn


    class AutogradTesting:
        def __init__(self):
            self.assert_allclose = _np.testing.assert_allclose
            self.assert_equal = _np.testing.assert_equal
