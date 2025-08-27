import numpy as np

import pytest

from sklearn.exceptions import NotFittedError

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

def test_attributes():
    n, sz, d = 5, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    labels = rng.randint(low=0, high=2, size=n)

    for ModelClass in [TimeSeriesSVC, TimeSeriesSVR]:
        linear_model = ModelClass(kernel="linear")

        for attr in ['coef_', 'support_', 'support_vectors_',
                     'dual_coef_', 'coef_', 'intercept_']:
            with pytest.raises(NotFittedError):
                getattr(linear_model, attr)

        linear_model.fit(time_series, labels)
        for attr in ['coef_', 'support_', 'support_vectors_',
                     'dual_coef_', 'coef_', 'intercept_']:
            assert hasattr(linear_model, attr)


def test_invalid_gamma_value():
    # Gamma computed with same time series is 0
    X = ([np.ones(3), np.ones(3)])
    y = [0, 1]

    for cls in [TimeSeriesSVC, TimeSeriesSVR]:
        estimator = cls(kernel="gak")
        with pytest.raises(RuntimeError):
            estimator.fit(X, y)

    # Explicit gamma set to 0
    n, sz, d = 5, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    labels = rng.randint(low=0, high=2, size=n)

    for cls in [TimeSeriesSVC, TimeSeriesSVR]:
        estimator = cls(kernel="gak", gamma=0)
        with pytest.raises(RuntimeError):
            estimator.fit(time_series, labels)
