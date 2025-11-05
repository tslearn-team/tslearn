from contextlib import contextmanager
from functools import partial
import textwrap
import warnings

import numpy as np

from sklearn.base import clone, is_regressor, is_classifier
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.utils import shuffle
from sklearn.utils._testing import (
    ignore_warnings,
    set_random_state,
    raises,
    assert_array_equal,
    assert_array_almost_equal,
    assert_allclose
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
        return getattr(get_tags(estimator), tag_name, None)
    except ImportError:
        return estimator._get_tags().get(tag_name)


@ignore_warnings(category=FutureWarning)
def check_regressors_train(
    name, regressor_orig, readonly_memmap=False, X_dtype=np.float64
):
    import sklearn.utils.estimator_checks as checks
    with patch(checks, '_regression_dataset', create_large_ts_dataset):
        checks.check_regressors_train(
            name,
            regressor_orig,
            readonly_memmap,
            X_dtype
        )


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
    if get_tag(clusterer, "non_deterministic"): # pragma: no cover
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


@ignore_warnings(category=FutureWarning)
def check_n_features_in_after_fitting(name, estimator_orig):
    from sklearn.utils import get_tags

    # Make sure that n_features_in are checked after fitting
    tags = get_tags(estimator_orig)

    is_supported_X_types = tags.input_tags.two_d_array or tags.input_tags.categorical

    if not is_supported_X_types or tags.no_validation:
        return

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    n_ts, n_samples, n_features = (10, 7, 2)
    X = rng.normal(size=(n_ts, n_samples, n_features))
    X = _enforce_estimator_tags_X(estimator, X)

    if is_regressor(estimator):
        y = rng.normal(size=n_ts)
    else:
        y = rng.randint(low=0, high=2, size=n_ts)
    y = _enforce_estimator_tags_y(estimator, y)

    err_msg = (
        "`{name}.fit()` does not set the `n_features_in_` attribute. "
        "You might want to use `sklearn.utils.validation.validate_data` instead "
        "of `check_array` in `{name}.fit()` which takes care of setting the "
        "attribute.".format(name=name)
    )

    estimator.fit(X, y)
    assert hasattr(estimator, "n_features_in_"), err_msg
    assert estimator.n_features_in_ == X.shape[-1], err_msg

    # check methods will check n_features_in_
    check_methods = [
        "predict",
        "transform",
        "decision_function",
        "predict_proba",
        "score",
    ]
    bad_shape = X.shape[:-1] + (1,)
    X_bad = np.resize(X, bad_shape)

    err_msg = """\
        `{name}.{method}()` does not check for consistency between input number
        of features with {name}.fit(), via the `n_features_in_` attribute.
        You might want to use `sklearn.utils.validation.validate_data` instead
        of `check_array` in `{name}.fit()` and {name}.{method}()`. This can be done
        like the following:
        from sklearn.utils.validation import validate_data
        ...
        class MyEstimator(BaseEstimator):
            ...
            def fit(self, X, y):
                X, y = validate_data(self, X, y, ...)
                ...
                return self
            ...
            def {method}(self, X):
                X = validate_data(self, X, ..., reset=False)
                ...
            return X
    """
    err_msg = textwrap.dedent(err_msg)

    msg = "(Number of features of the provided timeseries)|(Dimensions of the provided timeseries)"
    for method in check_methods:
        if not hasattr(estimator, method):
            continue

        callable_method = getattr(estimator, method)
        if method == "score":
            callable_method = partial(callable_method, y=y)

        with raises(
                ValueError, match=msg, err_msg=err_msg.format(name=name, method=method)
        ):
            callable_method(X_bad)

    # partial_fit will check in the second call
    if not hasattr(estimator, "partial_fit"):
        return

    estimator = clone(estimator_orig)
    if is_classifier(estimator):
        estimator.partial_fit(X, y, classes=np.unique(y))
    else:
        estimator.partial_fit(X, y)
    assert estimator.n_features_in_ == X.shape[-1]

    with raises(ValueError, match=msg):
        estimator.partial_fit(X_bad, y)


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(
    name,
    classifier_orig,
    readonly_memmap=False,
    X_dtype="float64"
):
    # Case of shapelet models
    if name  in ('LearningShapelets', 'TimeSeriesMLPClassifier'):
        X_m, y_m = random_walk_blobs(n_ts_per_blob=50, n_blobs=3, random_state=1,
                             sz=20, noise_level=0.025, d=2)
    else:
        X_m, y_m = random_walk_blobs(n_ts_per_blob=5, n_blobs=3, random_state=1,
                             sz=7, noise_level=0.025, d=2)
    X_m = X_m.astype(X_dtype)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = TimeSeriesScalerMeanVariance().fit_transform(X_m)

    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    if readonly_memmap:
        X_m, y_m, X_b, y_b = create_memmap_backed_data([X_m, y_m, X_b, y_b])

    problems = [(X_b, y_b)]
    if (get_tag(classifier_orig, "binary_only") or
            getattr(
                get_tag(classifier_orig, "classifier_tags"),
                "multi_class", None)):
        problems.append((X_m, y_m))

    for X, y in problems:
        classes = np.unique(y)
        n_classes = len(classes)
        n_ts, n_samples, n_features = X.shape
        classifier = clone(classifier_orig)
        X = _enforce_estimator_tags_X(classifier, X)
        y = _enforce_estimator_tags_y(classifier, y)

        set_random_state(classifier)
        # raises error on malformed input for fit
        if not get_tag(classifier_orig, "no_validation"):
            with raises(
                ValueError,
                err_msg=(
                    f"The classifier {name} does not raise an error when "
                    "incorrect/malformed input data for fit is passed. The number "
                    "of training examples is not the same as the number of "
                    "labels. Perhaps use check_X_y in fit."
                ),
            ):
                classifier.fit(X, y[:-1])

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)

        assert y_pred.shape == (n_ts,)
        # training set performance
        if not (get_tag(classifier_orig, "poor_score") or
            getattr(
                get_tag(classifier_orig, "classifier_tags"),
                "poor_score", None)):
            assert accuracy_score(y, y_pred) > 0.83

        # raises error on malformed input for predict
        msg_pairwise = (
            "The classifier {} does not raise an error when shape of X in "
            " {} is not equal to (n_test_samples, n_training_samples)"
        )
        msg = (
            "The classifier {} does not raise an error when the number of "
            "features in {} is different from the number of features in "
            "fit."
        )

        if not get_tag(classifier_orig, "no_validation"):
            with raises(ValueError, err_msg=msg.format(name, "predict")):
                classifier.predict(X.T)
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    if (not get_tag(classifier_orig, "multioutput_only") or
                        getattr(
                            get_tag(classifier_orig, "target_tags"),
                            "single_output", None)):
                        assert decision.shape == (n_ts,)
                    else: # pragma: no cover
                        assert decision.shape == (n_ts, 1)
                    dec_pred = (decision.ravel() > 0).astype(int)
                    assert_array_equal(dec_pred, y_pred)
                else:
                    assert decision.shape == (n_ts, n_classes)
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if not get_tag(classifier_orig, "no_validation"):
                    with raises(
                        ValueError,
                        err_msg=msg.format(name, "decision_function"),
                    ):
                        classifier.decision_function(X.T)
            except NotImplementedError: # pragma: no cover
                pass

        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert y_prob.shape == (n_ts, n_classes)
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_array_almost_equal(np.sum(y_prob, axis=1), np.ones(n_ts))
            if not get_tag(classifier_orig, "no_validation"):
                with raises(
                    ValueError,
                    err_msg=msg.format(name, "predict_proba"),
                ):
                    classifier.predict_proba(X.T)
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))
