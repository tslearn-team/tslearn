"""
The :mod:`tslearn.testing_utils` module includes various utilities that can be used for testing.
"""

import pkgutil
import inspect
import tslearn
from sklearn.base import (BaseEstimator, ClassifierMixin, ClusterMixin,
                          RegressorMixin, TransformerMixin)
from sklearn.utils.estimator_checks import check_estimator

def get_estimators(type_filter='all'):
    """Return a list of classes that inherit from `sklearn.BaseEstimator`.
    
    Parameters
    ----------
    type_filter : str
        A value in ['all', 'classifier', 'transformer', 'cluster'] which 
        defines which type of estimators to retrieve

    Returns
    -------
    list 
        Collection of estimators of the type specified in `type_filter`
    """

    if type_filter not in ['all', 'classifier', 'transformer', 'cluster']:
        # TODO: make this exception more specific
        raise Exception("type_filter should be element of "\
                        "['all', 'classifier', 'transformer', 'cluster']")

    # Walk through all the packages from our base_path and
    # add all the classes to a list
    all_classes = []
    base_path = tslearn.__path__
    for _, name, _ in pkgutil.walk_packages(path=base_path,
                                            prefix='tslearn.'):
        module = __import__(name, fromlist="dummy")
        all_classes.extend(inspect.getmembers(module, inspect.isclass))

    print(all_classes)

    # Filter out those that are not a subclass of `sklearn.BaseEstimator`
    all_classes = [c for c in all_classes
                   if issubclass(c[1], BaseEstimator)]

    print(all_classes)
     
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
    return list(set(filtered_classes))

def check_all_estimators():
	for estimator in get_estimators('all'):
		check_estimator(estimator)


check_all_estimators()