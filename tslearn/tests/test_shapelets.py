import numpy as np

from tslearn.shapelets import ShapeletModel, SerializableShapeletModel

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_shapelets():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)

    raise NotImplementedError