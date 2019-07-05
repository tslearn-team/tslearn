"""
The :mod:`tslearn.testing_utils` module includes various utilities that can
be used for testing.
"""

import tslearn

import pkgutil
import inspect
from operator import itemgetter
from functools import partial

import sklearn
from sklearn.base import (BaseEstimator, ClassifierMixin, ClusterMixin,
                          RegressorMixin, TransformerMixin)
from sklearn.utils.estimator_checks import *

def _yield_clustering_checks(name, clusterer):
    yield check_clusterer_compute_labels_predict
    if name not in ('WardAgglomeration', "FeatureAgglomeration", "KShape"):
        # this is clustering on the features
        # let's not test that here.
        yield check_clustering
        yield partial(check_clustering, readonly_memmap=True)
        yield check_estimators_partial_fit_n_features

    # TODO: Create a test similar to check_clustering (check_ts_clustering)
    # TODO: that generates a simple timeseries example and tests whether
    # TODO: the clustering algorithm can handle it.
    yield check_non_transformer_estimators_n_iter

sklearn.utils.estimator_checks._yield_clustering_checks = _yield_clustering_checks

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
        if estimator[0] in ['GlobalAlignmentKernelKMeans', 'KNeighborsTimeSeriesClassifier']:
            print('SKIP')
            continue
        check_estimator(estimator[1])
        print('{} is sklearn compliant.'.format(estimator[0]))


test_all_estimators()