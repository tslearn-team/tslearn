"""
The :mod:`tslearn.testing_utils` module includes various utilities that can
be used for testing.
"""

import os

import tslearn
import pkgutil
import inspect
from functools import partial
from operator import itemgetter

import sklearn
from sklearn.base import (BaseEstimator, ClassifierMixin, ClusterMixin,
                          RegressorMixin, TransformerMixin)

try:
    # Most recent
    from sklearn.utils._testing import SkipTest
except ImportError:
    # Deprecated from sklearn v0.24 onwards
    from sklearn.utils.testing import SkipTest
from sklearn.exceptions import SkipTestWarning
from sklearn.utils.estimator_checks import (
    check_no_attributes_set_in_init,
    check_parameters_default_constructible
)

try:
    # sklearn version < 1.6
    from sklearn.utils.estimator_checks import _maybe_skip
except ImportError:
    # sklearn version >= 1.6
    from sklearn.utils.estimator_checks import _maybe_mark
    def _maybe_skip(estimator, check):
        return sklearn.utils.estimator_checks._maybe_mark(estimator,
                                                          check,
                                                          estimator._get_tags().get('_xfail_checks'),
                                                          "skip")[1]


from tslearn.neural_network import TimeSeriesMLPClassifier
from tslearn.shapelets import LearningShapelets
from tslearn.tests.sklearn_patches import (
                             check_clustering,
                             check_non_transf_est_n_iter,
                             check_fit_idempotent,
                             check_classifiers_classes,
                             check_classifiers_train,
                             check_estimators_pickle,
                             check_supervised_y_2d,
                             check_regressor_data_not_an_array,
                             check_classifier_data_not_an_array,
                             check_regressors_int_patched,
                             check_classifiers_cont_target,
                             check_pipeline_consistency,
                             yield_all_checks,
                             _create_large_ts_dataset)

import warnings
import pytest


# Patching some check functions to work on ts data instead of tabular data.
checks = sklearn.utils.estimator_checks
checks._yield_all_checks = yield_all_checks
checks.check_clustering = check_clustering
checks.check_non_transformer_estimators_n_iter = check_non_transf_est_n_iter
checks.check_fit_idempotent = check_fit_idempotent
checks.check_classifiers_classes = check_classifiers_classes
checks.check_classifiers_train = check_classifiers_train
checks.check_estimators_pickle = check_estimators_pickle
checks.check_supervised_y_2d = check_supervised_y_2d
checks.check_regressor_data_not_an_array = check_regressor_data_not_an_array
checks.check_classifier_data_not_an_array = check_classifier_data_not_an_array
checks.check_regressors_int = check_regressors_int_patched
checks.check_classifiers_regression_target = check_classifiers_cont_target
checks.check_pipeline_consistency = check_pipeline_consistency
checks._regression_dataset = _create_large_ts_dataset


def _get_all_classes():
    # Walk through all the packages from our base_path and
    # add all the classes to a list
    all_classes = []
    base_path = tslearn.__path__
    for _, name, _ in pkgutil.walk_packages(path=base_path,
                                            prefix='tslearn.'):
        try:
            module = __import__(name, fromlist="dummy")
        except ImportError:
            if name.endswith('shapelets'):
                # keras is likely not installed
                warnings.warn('Skipped common tests for shapelets '
                              'as it could not be imported. keras '
                              '(and tensorflow) are probably not '
                              'installed!')
                continue
            elif name.endswith('pytorch_backend'):
                # pytorch is likely not installed
                continue
            else:
                raise Exception('Could not import module %s' % name)

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
    This code is based on `sklearn.utils.testing.all_estimators`.

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
    all_classes = filter(lambda c: not is_abstract(c[1]), all_classes)

    # only keep those that are from tslearn
    all_classes = filter(lambda c: not is_sklearn(c[1]), all_classes)

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


def check_estimator(Estimator):
    """Check if estimator adheres to scikit-learn conventions.
    This estimator will run an extensive test-suite for input validation,
    shapes, etc.
    Additional tests for classifiers, regressors, clustering or transformers
    will be run if the Estimator class inherits from the corresponding mixin
    from sklearn.base.
    This test can be applied to classes or instances.
    Classes currently have some additional tests that related to construction,
    while passing instances allows the testing of multiple options.
    Parameters
    ----------
    estimator : estimator object or class
        Estimator to check. Estimator is a class object or instance.
    """
    if isinstance(Estimator, type):
        # got a class
        name = Estimator.__name__
        estimator = Estimator()

        check_parameters_default_constructible(name, estimator)
        check_no_attributes_set_in_init(name, estimator)
    else:
        # got an instance
        estimator = Estimator
        name = type(estimator).__name__

    def checks_generator():
        for check in checks._yield_all_checks(name, estimator):
            check = _maybe_skip(estimator, check)
            yield estimator, partial(check, name)

    for estimator, check in checks_generator():
        try:
            _configure(estimator, check)
            check(estimator)
        except SkipTest as exception:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(str(exception), SkipTestWarning)


def _configure(estimator, check):
    """ Configure estimator for a given check depending on the platform """
    if hasattr(estimator, 'total_lengths'):
        estimator.set_params(total_lengths=1)

    if hasattr(estimator, 'probability'):
        estimator.set_params(probability=True)

    if (isinstance(estimator, (LearningShapelets,
                               TimeSeriesMLPClassifier)) and
        check.func.__name__ in ['check_classifiers_classes',
                                'check_classifiers_train']):
        estimator.set_params(max_iter=1000)
    elif isinstance(estimator, LearningShapelets):
        estimator.set_params(max_iter=100)
    elif hasattr(estimator, 'max_iter'):
        estimator.set_params(max_iter=10)

    if os.environ.get("SYSTEM_PHASENAME", "") == "codecov":
        # Tweak to ensure fast execution of code coverage job on azure pipelines
        from tslearn.shapelets.shapelets import _kmeans_init_shapelets
        _kmeans_init_shapelets.__defaults__ = (1,)
        if isinstance(estimator, (LearningShapelets, TimeSeriesMLPClassifier)):
            estimator._more_tags = lambda *args, **kwargs: {'poor_score': True,
                                                            'allow_nan': True,
                                                            'allow_variable_length': True}
            estimator.set_params(max_iter=1)


@pytest.mark.parametrize('name, Estimator', get_estimators('all'))
def test_all_estimators(name, Estimator):
    """Test all the estimators in tslearn."""
    allow_nan = (hasattr(checks, 'ALLOW_NAN') and
                 Estimator().get_tags()["allow_nan"])
    if allow_nan:
        checks.ALLOW_NAN.append(name)
    if name in ["ShapeletModel"]:
        # Deprecated models
        return
    check_estimator(Estimator)
