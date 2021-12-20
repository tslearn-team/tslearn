from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import sklearn
import warnings
from sklearn.base import clone

from sklearn.base import is_classifier, is_outlier_detector, is_regressor
from sklearn.base import ClusterMixin
from sklearn.exceptions import DataConversionWarning

from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.utils import shuffle
try:
    from sklearn.utils._testing import (
        set_random_state, assert_array_equal,
        assert_raises, assert_array_almost_equal,
        assert_allclose, assert_raises_regex, assert_allclose_dense_sparse
    )

    from unittest import TestCase
    _dummy = TestCase('__init__')
    assert_equal = _dummy.assertEqual
    assert_greater = _dummy.assertGreater
    assert_greater_equal = _dummy.assertGreaterEqual
except:
    from sklearn.utils.testing import (
        set_random_state, assert_equal, assert_greater, assert_array_equal,
        assert_raises, assert_array_almost_equal, assert_greater_equal,
        assert_allclose, assert_raises_regex, assert_allclose_dense_sparse
    )
    warnings.warn(
        "Scikit-learn <0.24 will be deprecated in a "
        "future release of tslearn"
    )

from sklearn.utils.estimator_checks import (
    check_classifiers_predictions,
    check_fit2d_1sample,
    check_fit2d_1feature,
    check_fit1d,
    check_get_params_invariance,
    check_set_params,
    check_dict_unchanged,
    check_dont_overwrite_parameters,
    check_estimators_data_not_an_array,
    check_fit2d_predict1d,
    check_methods_subset_invariance,
    check_regressors_int
)

try:
    # Most recent
    from sklearn.utils.estimator_checks import (
        _pairwise_estimator_convert_X as pairwise_estimator_convert_X,
        _choose_check_classifiers_labels as choose_check_classifiers_labels
    )
except ImportError:
    # Deprecated from sklearn v0.24 onwards
    from sklearn.utils.estimator_checks import (
        pairwise_estimator_convert_X,
        choose_check_classifiers_labels
    )

try:
    # Most recent
    from sklearn.utils._testing import ignore_warnings, SkipTest
except ImportError:
    # Deprecated from sklearn v0.24 onwards
    from sklearn.utils.testing import ignore_warnings, SkipTest
from sklearn.exceptions import SkipTestWarning
from sklearn.utils.estimator_checks import (_yield_classifier_checks,
                                            _yield_regressor_checks,
                                            _yield_transformer_checks,
                                            _yield_clustering_checks,
                                            _yield_outliers_checks)
try:
    from sklearn.utils.estimator_checks import _yield_checks
except ImportError:
    from sklearn.utils.estimator_checks import _yield_non_meta_checks
    _yield_checks = _yield_non_meta_checks

from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection._validation import _safe_split
from sklearn.pipeline import make_pipeline

from tslearn.clustering import TimeSeriesKMeans

import warnings
import numpy as np


def _create_small_ts_dataset():
    return random_walk_blobs(n_ts_per_blob=5, n_blobs=3, random_state=1,
                             sz=10, noise_level=0.025)


def _create_large_ts_dataset():
    return random_walk_blobs(n_ts_per_blob=50, n_blobs=3, random_state=1,
                             sz=20, noise_level=0.025)


def enforce_estimator_tags_y(estimator, y):
    # Estimators with a `requires_positive_y` tag only accept strictly positive
    # data
    if estimator._get_tags()["requires_positive_y"]:
        # Create strictly positive y. The minimal increment above 0 is 1, as
        # y could be of integer dtype.
        y += 1 + abs(y.min())
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if estimator._get_tags()["multioutput_only"]:
        return np.reshape(y, (-1, 1))
    return y


def multioutput_estimator_convert_y_2d(estimator, y):
    # This function seems to be removed in version 0.22, so let's make
    # a copy here.
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if "MultiTask" in estimator.__class__.__name__:
        return np.reshape(y, (-1, 1))
    return y


# Patch BOSTON dataset of sklearn to fix _csv.Error: line contains NULL byte
# Moreover, it makes more sense to use a timeseries dataset for our estimators
BOSTON = _create_small_ts_dataset()
sklearn.utils.estimator_checks.BOSTON = BOSTON


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_clustering(name, clusterer_orig, readonly_memmap=False):

    clusterer = clone(clusterer_orig)
    X, y = _create_small_ts_dataset()
    X, y = shuffle(X, y, random_state=7)
    X = TimeSeriesScalerMeanVariance().fit_transform(X)
    rng = np.random.RandomState(42)
    X_noise = X + (rng.randn(*X.shape) / 5)

    n_samples, n_features, dim = X.shape
    # catch deprecation and neighbors warnings
    if hasattr(clusterer, "n_clusters"):
        clusterer.set_params(n_clusters=3)
    set_random_state(clusterer)

    # fit
    clusterer.fit(X)
    # with lists
    clusterer.fit(X.tolist())

    pred = clusterer.labels_
    assert_equal(pred.shape, (n_samples,))
    assert_greater(adjusted_rand_score(pred, y), 0.4)

    if clusterer._get_tags()['non_deterministic']:
        return

    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert pred.dtype in [np.dtype('int32'), np.dtype('int64')]
    assert pred2.dtype in [np.dtype('int32'), np.dtype('int64')]

    # Add noise to X to test the possible values of the labels
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every original cluster
    labels_sorted = np.unique(labels)
    assert_array_equal(labels_sorted, np.arange(0, 3))

    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, 'n_clusters'):
        n_clusters = getattr(clusterer, 'n_clusters')
        assert_greater_equal(n_clusters - 1, labels_sorted[-1])

@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_non_transf_est_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.
    estimator = clone(estimator_orig)
    if hasattr(estimator, 'max_iter'):
        X, y = _create_small_ts_dataset()
        set_random_state(estimator, 0)
        estimator.fit(X, y)
        assert estimator.n_iter_ >= 1


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_fit_idempotent(name, estimator_orig):
    # Check that est.fit(X) is the same as est.fit(X).fit(X). Ideally we would
    # check that the estimated parameters during training (e.g. coefs_) are
    # the same, but having a universal comparison function for those
    # attributes is difficult and full of edge cases. So instead we check that
    # predict(), predict_proba(), decision_function() and transform() return
    # the same results.

    check_methods = ["predict", "transform", "decision_function",
                     "predict_proba"]
    rng = np.random.RandomState(0)

    if estimator_orig._get_tags()['non_deterministic']:
        msg = name + ' is non deterministic'
        raise SkipTest(msg)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if 'warm_start' in estimator.get_params().keys():
        estimator.set_params(warm_start=False)

    n_samples = 100
    X, _ = _create_small_ts_dataset()
    X = X.reshape((X.shape[0], X.shape[1]))
    X = pairwise_estimator_convert_X(X, estimator)
    if is_regressor(estimator_orig):
        y = rng.normal(size=n_samples)
    else:
        y = rng.randint(low=0, high=2, size=n_samples)

    train, test = next(ShuffleSplit(test_size=.2, random_state=rng).split(X))
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # Fit for the first time
    estimator.fit(X_train, y_train)

    result = {method: getattr(estimator, method)(X_test)
              for method in check_methods
              if hasattr(estimator, method)}

    # Fit again
    set_random_state(estimator)
    estimator.fit(X_train, y_train)

    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)
            if np.issubdtype(new_result.dtype, np.floating):
                tol = 2*np.finfo(new_result.dtype).eps
            else:
                tol = 2*np.finfo(np.float64).eps
            assert_allclose_dense_sparse(
                result[method], new_result,
                atol=max(tol, 1e-9), rtol=max(tol, 1e-7),
                err_msg="Idempotency check failed for method {}".format(method)
            )


def check_classifiers_classes(name, classifier_orig):
    # Case of shapelet models
    if name in ['LearningShapelets', 'TimeSeriesMLPClassifier']:
        X_multiclass, y_multiclass = _create_large_ts_dataset()
        classifier_orig = clone(classifier_orig)
        classifier_orig.max_iter = 1000
    else:
        X_multiclass, y_multiclass = _create_small_ts_dataset()

    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass,
                                         random_state=7)

    scaler = TimeSeriesScalerMeanVariance()
    X_multiclass = scaler.fit_transform(X_multiclass)

    X_multiclass = np.reshape(X_multiclass, (X_multiclass.shape[0],
                                             X_multiclass.shape[1]))

    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    X_multiclass = pairwise_estimator_convert_X(X_multiclass, classifier_orig)
    X_binary = pairwise_estimator_convert_X(X_binary, classifier_orig)

    labels_multiclass = ["one", "two", "three"]
    labels_binary = ["one", "two"]

    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)

    problems = [(X_binary, y_binary, y_names_binary)]

    if not classifier_orig._get_tags()['binary_only']:
        problems.append((X_multiclass, y_multiclass, y_names_multiclass))

    for X, y, y_names in problems:
        for y_names_i in [y_names, y_names.astype('O')]:
            y_ = choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)

    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig, readonly_memmap=False,
                            X_dtype='float64'):
    # Case of shapelet models
    if name in ['LearningShapelets', 'TimeSeriesMLPClassifier']:
        X_m, y_m = _create_large_ts_dataset()
        classifier_orig = clone(classifier_orig)
        classifier_orig.max_iter = 1000
    else:
        X_m, y_m = _create_small_ts_dataset()
    X_m = X_m.astype(X_dtype)

    X_m, y_m = shuffle(X_m, y_m, random_state=7)

    X_m = TimeSeriesScalerMeanVariance().fit_transform(X_m)

    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    # We will test for both binary and multiclass case
    problems = [(X_b, y_b), (X_m, y_m)]

    tags = classifier_orig._get_tags()

    for (X, y) in problems:
        classes = np.unique(y)
        n_classes = len(classes)
        n_samples, n_features, dim = X.shape
        classifier = clone(classifier_orig)
        X = pairwise_estimator_convert_X(X, classifier)

        set_random_state(classifier)
        # raises error on malformed input for fit
        if not tags["no_validation"]:
            with assert_raises(
                ValueError,
                msg="The classifier {} does not "
                    "raise an error when incorrect/malformed input "
                    "data for fit is passed. The number of training "
                    "examples is not the same as the number of labels. "
                    "Perhaps use check_X_y in fit.".format(name)):
                classifier.fit(X, y[:-1])

        # fit with lists
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)

        assert y_pred.shape == (n_samples,)

        # training set performance
        if not tags['poor_score']:
            assert accuracy_score(y, y_pred) > 0.83

        # raises error on malformed input for predict
        msg_pairwise = (
            "The classifier {} does not raise an error when shape of X in "
            " {} is not equal to (n_test_samples, n_training_samples)")
        msg = ("The classifier {} does not raise an error when the number of "
               "features in {} is different from the number of features in "
               "fit.")

        if not tags["no_validation"]:
            if bool(getattr(classifier, "_pairwise", False)):
                with assert_raises(ValueError,
                                   msg=msg_pairwise.format(name, "predict")):
                    classifier.predict(X.reshape(-1, 1))
            else:
                if not tags["allow_variable_length"]:
                    with assert_raises(ValueError,
                                       msg=msg.format(name, "predict")):
                        classifier.predict(X.T)
                else:
                    with assert_raises(ValueError,
                                       msg=msg.format(name, "predict")):
                        classifier.predict(X.reshape((-1, 5, 2)))
        if hasattr(classifier, "decision_function"):
            try:
                # decision_function agrees with predict
                decision = classifier.decision_function(X)
                if n_classes == 2:
                    if not tags["multioutput_only"]:
                        assert decision.shape == (n_samples,)
                    else:
                        assert decision.shape == (n_samples, 1)
                    dec_pred = (decision.ravel() > 0).astype(np.int)
                    assert_array_equal(dec_pred, y_pred)
                else:
                    assert decision.shape == (n_samples, n_classes)
                    assert_array_equal(np.argmax(decision, axis=1), y_pred)

                # raises error on malformed input for decision_function
                if not tags["no_validation"]:
                    error_msg = msg_pairwise.format(name, "decision_function")
                    if bool(getattr(classifier, "_pairwise", False)):
                        with assert_raises(ValueError, msg=error_msg):
                            classifier.decision_function(X.reshape(-1, 1))
                    else:
                        if not tags["allow_variable_length"]:
                            with assert_raises(ValueError, msg=error_msg):
                                classifier.decision_function(X.T)
                        else:
                            with assert_raises(ValueError, msg=error_msg):
                                classifier.decision_function(
                                    X.reshape((-1, 5, 2))
                                )
            except NotImplementedError:
                pass

        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            assert y_prob.shape == (n_samples, n_classes)
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_array_almost_equal(np.sum(y_prob, axis=1),
                                      np.ones(n_samples))
            if not tags["no_validation"]:
                # raises error on malformed input for predict_proba
                if bool(getattr(classifier_orig, "_pairwise", False)):
                    with assert_raises(ValueError, msg=msg_pairwise.format(
                            name, "predict_proba")):
                        classifier.predict_proba(X.reshape(-1, 1))
                else:
                    if not tags["allow_variable_length"]:
                        with assert_raises(ValueError, msg=msg.format(
                                name, "predict_proba")):
                            classifier.predict_proba(X.T)
                    else:
                        with assert_raises(ValueError, msg=msg.format(
                                name, "predict_proba")):
                            classifier.predict_proba(
                                X.reshape((-1, 5, 2))
                            )
            if hasattr(classifier, "predict_log_proba"):
                # predict_log_proba is a transformation of predict_proba
                y_log_prob = classifier.predict_log_proba(X)
                assert_allclose(y_log_prob, np.log(y_prob), 8, atol=1e-9)
                assert_array_equal(np.argsort(y_log_prob), np.argsort(y_prob))


@ignore_warnings
def check_estimators_pickle(name, estimator_orig):
    warnings.warn('Pickling is currently NOT tested!')
    pass


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_supervised_y_2d(name, estimator_orig):
    tags = estimator_orig._get_tags()
    X, y = _create_small_ts_dataset()
    if tags['binary_only']:
        X = X[y != 2]
        y = y[y != 2]

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    # fit
    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    set_random_state(estimator)
    # Check that when a 2D y is given, a DataConversionWarning is
    # raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DataConversionWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator.fit(X, y[:, np.newaxis])
    y_pred_2d = estimator.predict(X)
    msg = "expected 1 DataConversionWarning, got: %s" % (
        ", ".join([str(w_x) for w_x in w]))

    if not tags['multioutput'] and name not in ['TimeSeriesSVR']:
        # check that we warned if we don't support multi-output
        assert len(w) > 0, msg
        assert "DataConversionWarning('A column-vector y" \
               " was passed when a 1d array was expected" in msg
        assert_allclose(y_pred.ravel(), y_pred_2d.ravel())

@ignore_warnings(category=FutureWarning)
def check_classifier_data_not_an_array(name, estimator_orig):
    X, y = _create_large_ts_dataset()
    y = enforce_estimator_tags_y(estimator_orig, y)
    for obj_type in ["NotAnArray", "PandasDataframe"]:
        if obj_type == "PandasDataframe":
            X_ = X[:, :, 0]  # pandas df cant be 3d
        else:
            X_ = X
        check_estimators_data_not_an_array(name, estimator_orig, X_, y,
                                           obj_type)


@ignore_warnings(category=DeprecationWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    if name in ['TimeSeriesSVR']:
        return
    X, y = BOSTON
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = enforce_estimator_tags_y(estimator_orig, y)
    for obj_type in ["NotAnArray", "PandasDataframe"]:
        if obj_type == "PandasDataframe":
            X_ = X[:, :, 0]  # pandas df cant be 3d
        else:
            X_ = X
        check_estimators_data_not_an_array(name, estimator_orig, X_, y,
                                           obj_type)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_regressors_int_patched(name, regressor_orig):
    if name in ['TimeSeriesSVR']:
        return

    check_regressors_int(name, regressor_orig)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_classifiers_cont_target(name, estimator_orig):
    # Check if classifier throws an exception when fed regression targets

    X, _ = _create_small_ts_dataset()
    y = np.random.random(len(X))
    e = clone(estimator_orig)
    msg = 'Unknown label type: '
    if not e._get_tags()["no_validation"]:
        assert_raises_regex(ValueError, msg, e.fit, X, y)


@ignore_warnings
def check_pipeline_consistency(name, estimator_orig):
    if estimator_orig._get_tags()['non_deterministic']:
        msg = name + ' is non deterministic'
        raise SkipTest(msg)

    # check that make_pipeline(est) gives same score as est
    X, y = make_blobs(n_samples=30, centers=[[0, 0, 0], [1, 1, 1]],
                      random_state=0, n_features=2, cluster_std=0.1)
    X -= X.min()
    X = pairwise_estimator_convert_X(X, estimator_orig, kernel=rbf_kernel)
    estimator = clone(estimator_orig)
    y = multioutput_estimator_convert_y_2d(estimator, y)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)
    estimator.fit(X, y)
    pipeline.fit(X, y)

    funcs = ["score", "fit_transform"]

    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func_pipeline = getattr(pipeline, func_name)
            result = func(X, y)
            result_pipe = func_pipeline(X, y)
            assert_allclose_dense_sparse(result, result_pipe)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_different_length_fit_predict_transform(name, estimator):
    # Check if classifier can predict a dataset that does not have the same
    # number of timestamps as the data passed at fit time
    X, y = _create_small_ts_dataset()

    # Default for kmeans is Euclidean
    if isinstance(estimator, TimeSeriesKMeans):
        new_estimator = clone(estimator)
        new_estimator.metric = "dtw"
    elif name == "LearningShapelets":
        # Prepare shapelet model for long series
        new_estimator = clone(estimator)
        new_estimator.max_size = 2 * X.shape[1]
    else:
        new_estimator = estimator

    new_estimator.fit(X, y)

    X2 = np.hstack((X, X))
    X3 = np.stack((X[:, :, 0], X[:, :, 0]), axis=-1)
    check_methods = ["predict", "transform", "decision_function",
                     "predict_proba"]
    for func_name in check_methods:
        func = getattr(estimator, func_name, None)
        if func is not None:
            method = getattr(new_estimator, func_name)
            method(X2)

            with assert_raises(
                ValueError,
                msg="The estimator {} does not raise an error when number of "
                    "features (last dimension) is different between "
                    "fit and {}.".format(name, func_name)):
                method(X3)


def yield_all_checks(name, estimator):
    tags = estimator._get_tags()
    if "2darray" not in tags["X_types"]:
        warnings.warn("Can't test estimator {} which requires input "
                      " of type {}".format(name, tags["X_types"]),
                      SkipTestWarning)
        return
    if tags["_skip_test"]:
        warnings.warn("Explicit SKIP via _skip_test tag for estimator "
                      "{}.".format(name),
                      SkipTestWarning)
        return

    yield from _yield_checks(estimator)
    if is_classifier(estimator):
        yield from _yield_classifier_checks(estimator)
    if is_regressor(estimator):
        yield from _yield_regressor_checks(estimator)
    if hasattr(estimator, 'transform'):
        if not tags["allow_variable_length"]:
            # Transformer tests ensure that shapes are the same at fit and
            # transform time, hence we need to skip them for estimators that
            # allow variable-length inputs
            yield from _yield_transformer_checks(estimator)
    if isinstance(estimator, ClusterMixin):
        yield from _yield_clustering_checks(estimator)
    if is_outlier_detector(estimator):
        yield from _yield_outliers_checks(estimator)
    # We are not strict on presence/absence of the 3rd dimension
    # yield check_fit2d_predict1d

    if not tags["non_deterministic"]:
        yield check_methods_subset_invariance

    yield check_fit2d_1sample
    yield check_fit2d_1feature
    yield check_fit1d
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters
    yield check_fit_idempotent

    if (is_classifier(estimator) or
            is_regressor(estimator) or
            isinstance(estimator, ClusterMixin)):
        if tags["allow_variable_length"]:
            yield check_different_length_fit_predict_transform
