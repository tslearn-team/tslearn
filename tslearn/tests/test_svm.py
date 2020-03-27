import numpy as np

from tslearn.metrics import cdist_gak
from tslearn.svm import TimeSeriesSVC

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_svm_gak():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    labels = rng.randint(low=0, high=2, size=n)

    gamma = 10.
    gak_km = TimeSeriesSVC(kernel="gak", gamma=gamma)
    sklearn_X, _ = gak_km._preprocess_sklearn(time_series, labels,
                                              fit_time=True)

    cdist_mat = cdist_gak(time_series, sigma=np.sqrt(gamma / 2.))

    np.testing.assert_allclose(sklearn_X, cdist_mat)
