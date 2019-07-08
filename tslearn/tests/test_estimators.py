"""
The :mod:`tslearn.testing_utils` module includes various utilities that can
be used for testing.
"""

import tslearn

import pkgutil
import inspect
from operator import itemgetter
from functools import partial

from tslearn.generators import random_walks, random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import sklearn
from sklearn.base import (BaseEstimator, ClassifierMixin, ClusterMixin,
                          RegressorMixin, TransformerMixin, clone)
from sklearn.utils.testing import *
from sklearn.utils.estimator_checks import *
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection._validation import _safe_split
from sklearn.pipeline import make_pipeline

import numpy as np

import warnings


@ignore_warnings(category=(DeprecationWarning, FutureWarning))
def check_clustering(name, clusterer_orig, readonly_memmap=False):

    if hasattr(estimator_orig, '_get_tags'):
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

    # These models are dependent on external solvers like
    # libsvm and accessing the iter parameter is non-trivial.
    not_run_check_n_iter = ['Ridge', 'SVR', 'NuSVR', 'NuSVC',
                            'RidgeClassifier', 'SVC', 'RandomizedLasso',
                            'LogisticRegressionCV', 'LinearSVC',
                            'LogisticRegression']

    # Tested in test_transformer_n_iter
    not_run_check_n_iter += CROSS_DECOMPOSITION
    if name in not_run_check_n_iter:
        return

    # LassoLars stops early for the default alpha=1.0 the iris dataset.
    if name == 'LassoLars':
        estimator = clone(estimator_orig).set_params(alpha=0.)
    else:
        estimator = clone(estimator_orig)
    if hasattr(estimator, 'max_iter'):
        X, y_ = random_walk_blobs(n_ts_per_blob=15, n_blobs=3, random_state=1,
                                  noise_level=0.25)

        set_random_state(estimator, 0)
        if name == 'AffinityPropagation':
            estimator.fit(X)
        else:
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
    X_m, y_m = random_walk_blobs(n_ts_per_blob=100, random_state=0,
                                 n_blobs=3, noise_level=0., sz=200)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = TimeSeriesScalerMeanVariance().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]

    if name in ['BernoulliNB', 'MultinomialNB', 'ComplementNB']:
        X_m -= X_m.min()
        X_b -= X_b.min()

    problems = [(X_b, y_b)]
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

        # fit
        classifier.fit(X, y)
        # with lists
        classifier.fit(X.tolist(), y.tolist())
        assert hasattr(classifier, "classes_")
        y_pred = classifier.predict(X)

        assert y_pred.shape == (n_samples,)
        # training set performance
        if not tags['poor_score']:
            print(accuracy_score(y, y_pred))
            print(y)
            print(y_pred)
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
                    if _is_pairwise(classifier):
                        with assert_raises(ValueError, msg=msg_pairwise.format(
                                name, "decision_function")):
                            classifier.decision_function(X.reshape(-1, 1))
                    else:
                        with assert_raises(ValueError, msg=msg.format(
                                name, "decision_function")):
                            classifier.decision_function(X.T)
            except NotImplementedError:
                pass

        if hasattr(classifier, "predict_proba"):
            # predict_proba agrees with predict
            y_prob = classifier.predict_proba(X)
            print(y, y_prob, y_prob.shape)
            assert y_prob.shape == (n_samples, n_classes)
            assert_array_equal(np.argmax(y_prob, axis=1), y_pred)
            # check that probas for all classes sum to one
            assert_array_almost_equal(np.sum(y_prob, axis=1),
                                      np.ones(n_samples))
            if not tags["no_validation"]:
                # raises error on malformed input for predict_proba
                if _is_pairwise(classifier_orig):
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

# Patching some checks function to work on ts data instead of tabular data.
checks = sklearn.utils.estimator_checks
checks.check_clustering = check_clustering
checks.check_non_transformer_estimators_n_iter = check_non_transf_est_n_iter
checks.check_fit_idempotent = check_fit_idempotent
checks.check_classifiers_classes = check_classifiers_classes
checks.check_classifiers_train = check_classifiers_train


def get_estimators(type_filter='all'):
    """Return a list of classes that inherit from `sklearn.BaseEstimator`.
    This code is based on `sklearn,utils.testing.all_estimators`.

    Parameters
    ----------
    type_filter : str (default: 'all')
        A value in ['all', 'classifier', 'transformer', 'cluster'] which
        defines which type of estimators to retrieve

    Returns
    -------
    list
        Collection of estimators of the type specified in `type_filter`
    """

    def is_abstract(c):
        if not(hasattr(c, '__abstractmethods__')):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    if type_filter not in ['all', 'classifier', 'transformer', 'cluster']:
        # TODO: make this exception more specific
        raise Exception("type_filter should be element of "
                        "['all', 'classifier', 'transformer', 'cluster']")

    # Walk through all the packages from our base_path and
    # add all the classes to a list
    all_classes = []
    base_path = tslearn.__path__
    for _, name, _ in pkgutil.walk_packages(path=base_path,
                                            prefix='tslearn.'):
        module = __import__(name, fromlist="dummy")
        all_classes.extend(inspect.getmembers(module, inspect.isclass))

    # Filter out those that are not a subclass of `sklearn.BaseEstimator`
    all_classes = [c for c in set(all_classes)
                   if issubclass(c[1], BaseEstimator)]

    # get rid of abstract base classes
    all_classes = [c for c in all_classes if not is_abstract(c[1])]

    # only keep those that are from tslearn
    is_sklearn = lambda x: inspect.getmodule(x).__name__.startswith('sklearn')
    all_classes = [c for c in all_classes if not is_sklearn(c[1])]

    # Now filter out the estimators that are not of the specified type
    filters = {
        'all': [ClassifierMixin, RegressorMixin,
                TransformerMixin, ClusterMixin],
        'classifier': [ClassifierMixin],
        'transformer': [TransformerMixin],
        'cluster': [ClusterMixin]
    }[type_filter]
    filtered_classes = []
    for _class in all_classes:
        if any([issubclass(_class[1], mixin) for mixin in filters]):
            filtered_classes.append(_class)

    # Remove duplicates and return the list of remaining estimators
    return sorted(set(filtered_classes), key=itemgetter(0))


def test_all_estimators():
    estimators = get_estimators('all')
    for estimator in estimators:
        print(estimator[0])
        # TODO: This needs to be removed!!! (Currently here for faster testing)
        if estimator[0] in ['KNeighborsTimeSeriesClassifier',
                            'GlobalAlignmentKernelKMeans', 'KShape',
                            'LabelBinarizer', 'LabelCategorizer']:
            print('SKIPPED')
            continue

        check_estimator(estimator[1])
        print('{} is sklearn compliant.'.format(estimator[0]))


test_all_estimators()