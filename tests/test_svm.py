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
        sklearn_X, *_ = gak_model._preprocess_sklearn(time_series,
                                                     labels,
                                                     fit_time=True)

        cdist_mat = cdist_gak(time_series, sigma=np.sqrt(gamma / 2.))
        np.testing.assert_allclose(sklearn_X, cdist_mat)

    # Invalid gamma 0 for gak kernel
    for cls in [TimeSeriesSVC, TimeSeriesSVR]:
        estimator = cls(kernel="gak", gamma=0)
        with pytest.raises(RuntimeError):
            estimator.fit(time_series, labels)

    # Invalid computed gamma for gak kernel
    X = ([np.ones(3), np.ones(3)])
    y = [0, 1]

    for cls in [TimeSeriesSVC, TimeSeriesSVR]:
        estimator = cls(kernel="gak")
        with pytest.raises(RuntimeError):
            estimator.fit(X, y)


def test_gak_svm_long_time_series():
    # Non-regression test for
    # https://github.com/tslearn-team/tslearn/issues/450
    # The GAK kernel matrix used to be filled with NaN for time series longer
    # than about 405 samples, which made both estimators fail at fit time with
    # "ValueError: Input X contains NaN".
    n, sz, d = 6, 500, 1
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    labels = np.array([0, 0, 0, 1, 1, 1])

    cdist_mat = cdist_gak(time_series, sigma=1.0)
    assert np.isfinite(cdist_mat).all()

    for ModelClass in [TimeSeriesSVC, TimeSeriesSVR]:
        estimator = ModelClass(kernel="gak")
        estimator.fit(time_series, labels)
        assert np.isfinite(estimator.predict(time_series)).all()


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
