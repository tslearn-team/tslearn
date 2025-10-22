from contextlib import contextmanager
import warnings

import numpy as np

from sklearn.base import clone, is_regressor
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import shuffle
from sklearn.utils._testing import (
    ignore_warnings,
    set_random_state,
    raises,
    assert_array_equal,
)
from sklearn.utils.estimator_checks import (
    _enforce_estimator_tags_X,
    _enforce_estimator_tags_y,
    create_memmap_backed_data,
    check_transformer_general as sk_check_transformer_general
)

from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def create_small_ts_dataset():
    return random_walk_blobs(n_ts_per_blob=5, n_blobs=3, random_state=1,
                             sz=10, noise_level=0.025)


def create_large_ts_dataset():
    return random_walk_blobs(n_ts_per_blob=50, n_blobs=3, random_state=1,
                             sz=20, noise_level=0.025)


@contextmanager
def patch(module, name, patched):
    orig = getattr(module, name)
    setattr(module, name, patched)
    try:
        yield
    finally:
        setattr(module, name, orig)


def get_tag(estimator, tag_name):
    try:
        from sklearn.utils import get_tags
        return getattr(get_tags(estimator), tag_name)
    except ImportError:
        return estimator._get_tags().get(tag_name)


@ignore_warnings(category=FutureWarning)
def check_regressors_train(
    name, regressor_orig, readonly_memmap=False, X_dtype=np.float64
):
    import sklearn.utils.estimator_checks as checks
    with patch(checks, '_regression_dataset', create_large_ts_dataset):
        checks.check_regressors_train(name, regressor_orig, readonly_memmap, X_dtype)


def check_n_features_in(name, estimator_orig):
    # Make sure that n_features_in_ attribute doesn't exist until fit is
    # called, and that its value is correct.

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    n_ts, n_samples, n_features = (10, 7, 2)
    X = rng.normal(size=(n_ts, n_samples, n_features))
    X = _enforce_estimator_tags_X(estimator, X)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_ts)
    else:
        y = rng.randint(low=0, high=2, size=n_ts)
    y = _enforce_estimator_tags_y(estimator, y)

    assert not hasattr(estimator, "n_features_in_")
    estimator.fit(X, y)
    assert hasattr(estimator, "n_features_in_")
    assert estimator.n_features_in_ == X.shape[-1]


def check_transformer_general(name, estimator, readonly_memmap=False):
    if get_tag(estimator, ALLOW_VARIABLE_LENGTH):
        with raises(
            AssertionError,
            match=(
                f"The transformer {name} does not raise an error "
                "when the number of features in transform is different from "
                "the number of features in fit."
            ),
            err_msg="Estimators allowing variable length input should pass this test"
        ):
            sk_check_transformer_general(name, estimator, readonly_memmap)
    else:
        sk_check_transformer_general(name, estimator, readonly_memmap)


@ignore_warnings(category=FutureWarning)
def check_clustering(name, clusterer_orig, readonly_memmap=False):
    clusterer = clone(clusterer_orig)
    X, y = create_small_ts_dataset()
    X, y = shuffle(X, y, random_state=7)
    X = TimeSeriesScalerMeanVariance().fit_transform(X)
    rng = np.random.RandomState(7)
    X_noise = X + (rng.randn(*X.shape) / 5)

    if readonly_memmap:
        X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

    n_ts, n_samples, n_features = X.shape
    # catch deprecation and neighbors warnings
    if hasattr(clusterer, "n_clusters"):
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)

    # fit
    clusterer.fit(X)
    # with lists
    clusterer.fit(X.tolist())

    pred = clusterer.labels_
    assert pred.shape == (n_ts,)
    assert adjusted_rand_score(pred, y) > 0.4
    if get_tag(clusterer, "non_deterministic"):
        return
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert pred.dtype in [np.dtype("int32"), np.dtype("int64")]
    assert pred2.dtype in [np.dtype("int32"), np.dtype("int64")]

    # Add noise to X to test the possible values of the labels
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(
        labels_sorted, np.arange(labels_sorted[0], labels_sorted[-1] + 1)
    )

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert labels_sorted[0] in [0, -1]
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, "n_clusters"):
        n_clusters = getattr(clusterer, "n_clusters")
        assert n_clusters - 1 >= labels_sorted[-1]
    # else labels should be less than max(labels_) which is necessarily true
