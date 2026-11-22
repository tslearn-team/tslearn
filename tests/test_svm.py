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


def test_svr_epsilon_is_used():
    # TimeSeriesSVR accepted and documented `epsilon` but never passed it on
    # to the underlying sklearn SVR, so the epsilon-tube width was stuck at
    # the sklearn default whatever the user asked for.
    n, sz, d = 40, 20, 1
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    targets = time_series[:, :, 0].mean(axis=1) * 5.

    # `linear` keeps the fit deterministic (gak's gamma="auto" subsamples).
    small = TimeSeriesSVR(kernel="linear", epsilon=0.1).fit(time_series, targets)
    large = TimeSeriesSVR(kernel="linear", epsilon=5.).fit(time_series, targets)

    assert small.svm_estimator_.epsilon == 0.1
    assert large.svm_estimator_.epsilon == 5.

    # A tube that wide leaves no point outside it, hence no support vector.
    assert len(small.svm_estimator_.support_) > 0
    assert len(large.svm_estimator_.support_) == 0


# Characterization tests for fit (issue #1): they pin down what fit does today
# for both estimators, so moving it into TimeSeriesSVMMixin can't change it.

def _dataset(n=12, sz=10, d=3, seed=0):
    rng = np.random.RandomState(seed)
    time_series = rng.randn(n, sz, d)
    labels = np.array([0, 1] * (n // 2))
    return time_series, labels


def test_svc_fit_forwards_hyperparameters():
    time_series, labels = _dataset()
    model = TimeSeriesSVC(
        C=3., kernel="linear", degree=2, gamma=.5, coef0=.25, shrinking=False,
        probability=True, tol=1e-2, cache_size=100,
        class_weight={0: 1., 1: 2.}, verbose=False, max_iter=10000,
        decision_function_shape="ovo", random_state=3
    ).fit(time_series, labels)

    expected = {
        "C": 3., "kernel": "linear", "degree": 2, "gamma": .5, "coef0": .25,
        "shrinking": False, "probability": True, "tol": 1e-2,
        "cache_size": 100, "class_weight": {0: 1., 1: 2.}, "verbose": False,
        "max_iter": 10000, "decision_function_shape": "ovo", "random_state": 3
    }
    params = model.svm_estimator_.get_params()
    assert {key: params[key] for key in expected} == expected


def test_svr_fit_forwards_hyperparameters():
    time_series, labels = _dataset()
    model = TimeSeriesSVR(
        C=3., kernel="linear", degree=2, gamma=.5, coef0=.25, tol=1e-2,
        epsilon=.3, shrinking=False, cache_size=100, verbose=False,
        max_iter=10000
    ).fit(time_series, labels.astype(float))

    expected = {
        "C": 3., "kernel": "linear", "degree": 2, "gamma": .5, "coef0": .25,
        "tol": 1e-2, "epsilon": .3, "shrinking": False, "cache_size": 100,
        "verbose": False, "max_iter": 10000
    }
    params = model.svm_estimator_.get_params()
    assert {key: params[key] for key in expected} == expected


@pytest.mark.parametrize("ModelClass", [TimeSeriesSVC, TimeSeriesSVR])
def test_fit_gak_uses_precomputed_kernel(ModelClass):
    time_series, labels = _dataset()
    model = ModelClass(kernel="gak", gamma=10.).fit(time_series, labels)

    assert model.svm_estimator_.kernel == "precomputed"
    assert model.svm_estimator_.gamma == 10.


@pytest.mark.parametrize("ModelClass", [TimeSeriesSVC, TimeSeriesSVR])
def test_fit_returns_self_and_sets_n_features_in(ModelClass):
    time_series, labels = _dataset(d=3)
    model = ModelClass(kernel="linear")

    assert model.fit(time_series, labels) is model
    assert model.n_features_in_ == 3


@pytest.mark.parametrize("ModelClass", [TimeSeriesSVC, TimeSeriesSVR])
def test_fit_zero_sample_weight_drops_the_sample(ModelClass):
    time_series, labels = _dataset()
    weights = np.ones(len(labels))
    weights[:2] = 0.

    weighted = ModelClass(kernel="linear").fit(
        time_series, labels, sample_weight=weights)
    dropped = ModelClass(kernel="linear").fit(time_series[2:], labels[2:])

    np.testing.assert_allclose(weighted.predict(time_series),
                               dropped.predict(time_series))


@pytest.mark.parametrize("ModelClass", [TimeSeriesSVC, TimeSeriesSVR])
def test_fit_gamma_error_messages(ModelClass):
    time_series, labels = _dataset()
    with pytest.raises(RuntimeError,
                       match=r"^The `gamma` parameter is close to 0 and "
                             r"cannot be used with gak kernel\.$"):
        ModelClass(kernel="gak", gamma=0).fit(time_series, labels)

    constant = [np.ones(3), np.ones(3)]
    with pytest.raises(RuntimeError,
                       match=r"^The auto computed `gamma` parameter is close "
                             r"to 0 and cannot be used with gak kernel\.$"):
        ModelClass(kernel="gak").fit(constant, [0, 1])


def test_predictions_match_sklearn_on_the_same_kernel():
    # sklearn fitted on the flattened series (linear) or on the GAK Gram
    # matrix (gak) is an independent oracle for what fit + predict compute.
    from sklearn.svm import SVC, SVR

    time_series, labels = _dataset()
    targets = time_series[:, :, 0].mean(axis=1)
    flat = time_series.reshape((len(time_series), -1))
    gram = cdist_gak(time_series, sigma=np.sqrt(10. / 2.))

    for ModelClass, Sklearn, y in [(TimeSeriesSVC, SVC, labels),
                                   (TimeSeriesSVR, SVR, targets)]:
        ours = ModelClass(kernel="linear").fit(time_series, y)
        ref = Sklearn(kernel="linear").fit(flat, y)
        np.testing.assert_allclose(ours.predict(time_series),
                                   ref.predict(flat))

        ours = ModelClass(kernel="gak", gamma=10.).fit(time_series, y)
        ref = Sklearn(kernel="precomputed").fit(gram, y)
        np.testing.assert_allclose(ours.predict(time_series),
                                   ref.predict(gram))
