import numpy as np

from tslearn.metrics import cdist_gak
from tslearn.svm import TimeSeriesSVC, TimeSeriesSVR

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_gamma_value_svm():
    n, sz, d = 5, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    labels = rng.randint(low=0, high=2, size=n)

    gamma = 10.
    for ModelClass in [TimeSeriesSVC, TimeSeriesSVR]:
        gak_model = ModelClass(kernel="gak", gamma=gamma)
        sklearn_X, _ = gak_model._preprocess_sklearn(time_series,
                                                     labels,
                                                     fit_time=True)

        cdist_mat = cdist_gak(time_series, sigma=np.sqrt(gamma / 2.))

        np.testing.assert_allclose(sklearn_X, cdist_mat)


def test_deprecated_still_work():
    n, sz, d = 5, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    y = rng.randint(low=0, high=2, size=n)

    for ModelClass in [TimeSeriesSVC, TimeSeriesSVR]:
        clf = ModelClass().fit(X, y)
        np.testing.assert_equal(clf.support_vectors_time_series_().shape[1:],
                                X.shape[1:])
