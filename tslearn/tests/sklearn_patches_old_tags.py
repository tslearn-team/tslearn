from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.utils._testing import (
    assert_array_almost_equal,
    assert_allclose
)
from sklearn.utils.estimator_checks import (
    _is_pairwise_metric,
)

from tslearn.tests.sklearn_patches_base import *


from unittest import TestCase
_dummy = TestCase('__init__')
assert_equal = _dummy.assertEqual
assert_greater = _dummy.assertGreater
assert_greater_equal = _dummy.assertGreaterEqual
assert_raises = _dummy.assertRaises


def pairwise_estimator_convert_X(X, estimator, kernel=linear_kernel):
    if _is_pairwise_metric(estimator):
        return pairwise_distances(X, metric="euclidean")
    if estimator._get_tags()["pairwise"]:
        return kernel(X, X)

    return X


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
