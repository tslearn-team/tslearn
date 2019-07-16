"""
The :mod:`tslearn.testing_utils` module includes various utilities that can
be used for testing.
"""

import tslearn
import pkgutil
import inspect
from operator import itemgetter

import sklearn
from sklearn.base import (BaseEstimator, ClassifierMixin, ClusterMixin,
                          RegressorMixin, TransformerMixin)

from sklearn.utils.testing import *
from sklearn_patches import *

import warnings


# Patching some checks function to work on ts data instead of tabular data.
checks = sklearn.utils.estimator_checks
checks.check_clustering = check_clustering
checks.check_non_transformer_estimators_n_iter = check_non_transf_est_n_iter
checks.check_fit_idempotent = check_fit_idempotent
checks.check_classifiers_classes = check_classifiers_classes
checks.check_classifiers_train = check_classifiers_train
checks.check_estimators_pickle = check_estimators_pickle
checks.check_supervised_y_2d = check_supervised_y_2d
checks.check_regressor_data_not_an_array = check_regressor_data_not_an_array
checks.check_regressors_int = check_regressors_int_patched


def _get_all_classes():
    # Walk through all the packages from our base_path and
    # add all the classes to a list
    all_classes = []
    base_path = tslearn.__path__
    for _, name, _ in pkgutil.walk_packages(path=base_path,
                                            prefix='tslearn.'):
        module = __import__(name, fromlist="dummy")
        all_classes.extend(inspect.getmembers(module, inspect.isclass))
    return all_classes


def is_abstract(c):
    if not(hasattr(c, '__abstractmethods__')):
        return False
    if not len(c.__abstractmethods__):
        return False
    return True


def is_sklearn(x):
    return inspect.getmodule(x).__name__.startswith('sklearn')


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

    if type_filter not in ['all', 'classifier', 'transformer', 'cluster']:
        # TODO: make this exception more specific
        raise Exception("type_filter should be element of "
                        "['all', 'classifier', 'transformer', 'cluster']")

    all_classes = _get_all_classes()

    # Filter out those that are not a subclass of `sklearn.BaseEstimator`
    all_classes = [c for c in set(all_classes)
                   if issubclass(c[1], BaseEstimator)]

    # get rid of abstract base classes
    all_classes = [c for c in all_classes if not is_abstract(c[1])]

    # only keep those that are from tslearn
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
        # TODO: Remove what's below
        # if estimator[0] in ['KNeighborsTimeSeriesClassifier',
        #                     'GlobalAlignmentKernelKMeans', 'KShape',
        #                     'ShapeletModel', 'SerializableShapeletModel',
        #                     'LabelBinarizer', 'LabelCategorizer', 
        #                     'TimeSeriesKMeans', 'TimeSeriesSVC']: 
        #     print('SKIPPED')
        #     continue
        # TODO: Remove the above

        warnings.warn('Checking {}'.format(estimator[0]))
        check_estimator(estimator[1])
        warnings.warn('{} is sklearn compliant.'.format(estimator[0]))


# TODO: remove this
test_all_estimators()
