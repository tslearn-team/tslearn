import warnings

from sklearn.base import clone, is_regressor
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.utils import shuffle
from sklearn.utils._testing import (
    set_random_state,
    assert_array_equal,
    assert_array_almost_equal,
    assert_allclose,
    raises,
    SkipTest
)
from sklearn.utils.estimator_checks import (
    _enforce_estimator_tags_X,
    _enforce_estimator_tags_y,
    _is_pairwise_metric,
    check_estimators_data_not_an_array,
    check_transformer_general as sk_check_transformer_general,
)

from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.tests.sklearn_patches_base import *


from unittest import TestCase
_dummy = TestCase('__init__')
assert_equal = _dummy.assertEqual
assert_greater = _dummy.assertGreater
assert_greater_equal = _dummy.assertGreaterEqual
assert_raises = _dummy.assertRaises
assert_raises_regex = _dummy.assertRaisesRegex


def pairwise_estimator_convert_X(X, estimator, kernel=linear_kernel):
    if _is_pairwise_metric(estimator):
        return pairwise_distances(X, metric="euclidean")
    if estimator._get_tags()["pairwise"]:
        return kernel(X, X)

    return X


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_clustering(name, clusterer_orig, readonly_memmap=False):

    clusterer = clone(clusterer_orig)
    X, y = create_small_ts_dataset()
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


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig, readonly_memmap=False,
                            X_dtype='float64'):
    # Case of shapelet models
    if name in ['LearningShapelets', 'TimeSeriesMLPClassifier']:
        X_m, y_m = create_large_ts_dataset()
    else:
        X_m, y_m = create_small_ts_dataset()
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
                    dec_pred = (decision.ravel() > 0).astype(int)
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
def check_estimators_pickle(*args, **kwargs):
    raise SkipTest('Pickling is currently NOT tested!')


@ignore_warnings(category=DeprecationWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    X, y = create_small_ts_dataset()
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = _enforce_estimator_tags_y(estimator_orig, y)
    for obj_type in ["NotAnArray", "PandasDataframe"]:
        if obj_type == "PandasDataframe":
            X_ = X[:, :, 0]  # pandas df cant be 3d
        else:
            X_ = X
        check_estimators_data_not_an_array(name, estimator_orig, X_, y,
                                           obj_type)


def check_transformer_general(name, estimator, readonly_memmap=False):
    if estimator._get_tags()[ALLOW_VARIABLE_LENGTH]:
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


def check_n_features_in(name, estimator_orig):
    # Make sure that n_features_in_ attribute doesn't exist until fit is
    # called, and that its value is correct.

    rng = np.random.RandomState(0)

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if "warm_start" in estimator.get_params():
        estimator.set_params(warm_start=False)

    n_ts = 15
    n_samples = 10
    n_features = 3 if name != "MatrixProfile" else 1
    X = rng.normal(loc=100, size=(n_ts, n_samples, n_features))
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
