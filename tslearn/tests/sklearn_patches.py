from tslearn.generators import random_walks, random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import sklearn
from sklearn.base import clone
from sklearn.utils.testing import *
from sklearn.utils.estimator_checks import *
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection._validation import _safe_split
from sklearn.pipeline import make_pipeline

import warnings
import numpy as np


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_clustering(name, clusterer_orig, readonly_memmap=False):

    if hasattr(clusterer_orig, '_get_tags'):
        warnings.warn('Tags (_get_tags) are currently ignored by '
                      'check_clustering!')

    clusterer = clone(clusterer_orig)
    X, y = random_walk_blobs(n_ts_per_blob=15, n_blobs=3, random_state=1,
                             noise_level=0.25)
    X, y = shuffle(X, y, random_state=7)
    X = TimeSeriesScalerMeanVariance().fit_transform(X)
    X_noise = np.concatenate([X, random_walks(n_ts=5)])

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
    set_random_state(clusterer)
    with warnings.catch_warnings(record=True):
        pred2 = clusterer.fit_predict(X)
    assert_array_equal(pred, pred2)

    # fit_predict(X) and labels_ should be of type int
    assert_in(pred.dtype, [np.dtype('int32'), np.dtype('int64')])
    assert_in(pred2.dtype, [np.dtype('int32'), np.dtype('int64')])

    # Add noise to X to test the possible values of the labels
    labels = clusterer.fit_predict(X_noise)

    # There should be at least one sample in every cluster. Equivalently
    # labels_ should contain all the consecutive values between its
    # min and its max.
    labels_sorted = np.unique(labels)
    assert_array_equal(labels_sorted, np.arange(labels_sorted[0],
                                                labels_sorted[-1] + 1))

    # Labels are expected to start at 0 (no noise) or -1 (if noise)
    assert labels_sorted[0] in [0, -1]
    # Labels should be less than n_clusters - 1
    if hasattr(clusterer, 'n_clusters'):
        n_clusters = getattr(clusterer, 'n_clusters')
        assert_greater_equal(n_clusters - 1, labels_sorted[-1])
    # else labels should be less than max(labels_) which is necessarily true


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_non_transf_est_n_iter(name, estimator_orig):
    # Test that estimators that are not transformers with a parameter
    # max_iter, return the attribute of n_iter_ at least 1.
    if hasattr(estimator_orig, '_get_tags'):
        warnings.warn('Tags (_get_tags) are currently ignored by '
                      'check_non_transformer_estimators_n_iter!')
    estimator = clone(estimator_orig)
    if hasattr(estimator, 'max_iter'):
        X, y_ = random_walk_blobs(n_ts_per_blob=15, n_blobs=3, random_state=1,
                                  noise_level=0.25)
        set_random_state(estimator, 0)
        estimator.fit(X, y_)

        assert estimator.n_iter_ >= 1


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_fit_idempotent(name, estimator_orig):
    # Check that est.fit(X) is the same as est.fit(X).fit(X). Ideally we would
    # check that the estimated parameters during training (e.g. coefs_) are
    # the same, but having a universal comparison function for those
    # attributes is difficult and full of edge cases. So instead we check that
    # predict(), predict_proba(), decision_function() and transform() return
    # the same results.

    if hasattr(estimator_orig, '_get_tags'):
        warnings.warn('Tags (_get_tags) are currently ignored by '
                      'check_fit_idempotent!')

    check_methods = ["predict", "transform", "decision_function",
                     "predict_proba"]
    rng = np.random.RandomState(0)

    if name in ['TimeSeriesSVC', 'TimeSeriesSVR']:
        return

    estimator = clone(estimator_orig)
    set_random_state(estimator)
    if 'warm_start' in estimator.get_params().keys():
        estimator.set_params(warm_start=False)

    n_samples = 100
    X = random_walks(n_ts=n_samples)
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
    X_multiclass, y_multiclass = random_walk_blobs(n_ts_per_blob=10,
                                                   random_state=0,
                                                   n_blobs=3,
                                                   noise_level=0.1,
                                                   sz=100)

    X_multiclass, y_multiclass = shuffle(X_multiclass, y_multiclass,
                                         random_state=7)

    scaler = TimeSeriesScalerMeanVariance()
    X_multiclass = scaler.fit_transform(X_multiclass)

    X_binary = X_multiclass[y_multiclass != 2]
    y_binary = y_multiclass[y_multiclass != 2]

    X_multiclass = pairwise_estimator_convert_X(X_multiclass, classifier_orig)
    X_binary = pairwise_estimator_convert_X(X_binary, classifier_orig)

    labels_multiclass = ["one", "two", "three"]
    labels_binary = ["one", "two"]

    y_names_multiclass = np.take(labels_multiclass, y_multiclass)
    y_names_binary = np.take(labels_binary, y_binary)

    problems = [(X_binary, y_binary, y_names_binary)]

    if hasattr(classifier_orig, '_get_tags'):
        warnings.warn('Tags (_get_tags) are currently ignored by '
                      'check_classifiers_classes!')

    for X, y, y_names in problems:
        for y_names_i in [y_names, y_names.astype('O')]:
            y_ = choose_check_classifiers_labels(name, y, y_names_i)
            check_classifiers_predictions(X, y_, name, classifier_orig)

    labels_binary = [-1, 1]
    y_names_binary = np.take(labels_binary, y_binary)
    y_binary = choose_check_classifiers_labels(name, y_binary, y_names_binary)
    check_classifiers_predictions(X_binary, y_binary, name, classifier_orig)


@ignore_warnings  # Warnings are raised by decision function
def check_classifiers_train(name, classifier_orig, readonly_memmap=False):
    # Generate some random walk blobs, shuffle them and normalize them
    X_m, y_m = random_walk_blobs(n_ts_per_blob=25, random_state=42,
                                 n_blobs=3, noise_level=0.1, sz=75)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)

    X_m = TimeSeriesScalerMeanVariance().fit_transform(X_m)

    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    # We will test for both binary and multiclass case
    problems = [(X_b, y_b), (X_m, y_m)]

    if hasattr(classifier_orig, '_get_tags'):
        warnings.warn('Tags (_get_tags) are currently ignored by '
                      'check_classifiers_train!')
    tags = {'binary_only': False, 'no_validation': False,
            'poor_score': False, 'multioutput_only': False}

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
                with assert_raises(ValueError,
                                   msg=msg.format(name, "predict")):
                    classifier.predict(X.T)
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
                    if bool(getattr(classifier, "_pairwise", False)):
                        error_msg = msg_pairwise.format(name, "decision_function")
                        with assert_raises(ValueError, msg=error_msg):
                            classifier.decision_function(X.reshape(-1, 1))
                    else:
                        error_msg = msg_pairwise.format(name, "decision_function")
                        with assert_raises(ValueError, msg=error_msg):
                            classifier.decision_function(X.T)
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
                    with assert_raises(ValueError, msg=msg.format(
                            name, "predict_proba")):
                        classifier.predict_proba(X.T)
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
     
    tags = {'multioutput': False, 'binary_only': False}

    rnd = np.random.RandomState(0)
    X = random_walks(n_ts=10, sz=50)
    if tags['binary_only']:
        y = np.arange(10) % 2
    else:
        y = np.arange(10) % 3

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


@ignore_warnings(category=DeprecationWarning)
def check_regressor_data_not_an_array(name, estimator_orig):
    if name in ['TimeSeriesSVR']:
        return
    X, y = _boston_subset(n_samples=50)
    X = pairwise_estimator_convert_X(X, estimator_orig)
    y = enforce_estimator_tags_y(estimator_orig, y)
    check_estimators_data_not_an_array(name, estimator_orig, X, y)


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_regressors_int_patched(name, regressor_orig):
    if name in ['TimeSeriesSVR']:
        return

    check_regressors_int(name, regressor_orig)