"""
The :mod:`tslearn.testing_utils` module includes various utilities that can
be used for testing.
"""
from contextlib import contextmanager
from functools import partial
import inspect
import pkgutil
import os
import warnings

import sklearn
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin
)
from sklearn.utils.estimator_checks import parametrize_with_checks

import tslearn


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
            if name.endswith('sklearn_patches_new_tags') and sklearn.__version__ < '1.6':
                continue
            if name.endswith('sklearn_patches_old_tags') and sklearn.__version__ >= '1.6':
                continue
            if name.endswith('shapelets'):
                # keras is likely not installed
                warnings.warn('Skipped common tests for shapelets '
                              'as it could not be imported. keras '
                              'is probably not '
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
        raise ValueError("type_filter should be element of "
                        "['all', 'classifier', 'transformer', 'cluster']")

    all_classes = _get_all_classes()

    # Filter out those that are not a subclass of `sklearn.BaseEstimator`
    all_classes = [c[1] for c in set(all_classes)
                   if issubclass(c[1], BaseEstimator)]

    # get rid of abstract base classes
    all_classes = filter(lambda c: not is_abstract(c), all_classes)

    # only keep those that are from tslearn
    all_classes = filter(lambda c: not is_sklearn(c), all_classes)

    # Now filter out the estimators that are not of the specified type
    filters = {
        'all': [ClassifierMixin, RegressorMixin,
                TransformerMixin, ClusterMixin],
        'classifier': [ClassifierMixin],
        'transformer': [TransformerMixin],
        'cluster': [ClusterMixin]
    }[type_filter]
    filtered_estimators = []
    for _class in set(all_classes):
        if any([issubclass(_class, mixin) for mixin in filters]):
            filtered_estimators.append(_class())
            # Add variable length version of estimator
            if _class.__name__ in ["TimeSeriesKmeans"]:
                filtered_estimators.append(_class(metric="dtw"))
    return sorted(filtered_estimators, key=lambda x: x.__class__.__name__)


@contextmanager
def _configure(estimator, check):
    """ Configure estimator for a given check depending on the platform """
    if hasattr(estimator, 'total_lengths'):
        estimator.set_params(total_lengths=1)

    if hasattr(estimator, 'probability'):
        estimator.set_params(probability=True)

    if (estimator.__class__.__name__ in ("LearningShapelets",
                                         "TimeSeriesMLPClassifier") and
        check.func.__name__ in ['check_classifiers_classes',
                                'check_classifiers_train']):
        estimator.set_params(max_iter=1000)
    elif estimator.__class__.__name__ == "LearningShapelets":
        estimator.set_params(max_iter=100)
    elif hasattr(estimator, 'max_iter'):
        estimator.set_params(max_iter=10)

    if os.environ.get("SYSTEM_PHASENAME", "") == "codecov":
        # Tweak to ensure fast execution of code coverage job on azure pipelines
        from tslearn.shapelets.shapelets import _kmeans_init_shapelets
        _kmeans_init_shapelets.__defaults__ = (1,)
        if estimator.__class__.__name__ in ("LearningShapelets", "TimeSeriesMLPClassifier"):
            get_tags_orig = estimator._get_tags
            def get_tags_poor_score():
                tags = get_tags_orig()
                tags.update({"poor_score": True})
                return tags
            estimator._get_tags = get_tags_poor_score
            sklearn_tags_orig = estimator.__sklearn_tags__
            def sklearn_tags_poor_score():
                tags = sklearn_tags_orig()
                tags.classifier_tags.poor_score = True
                return tags
            estimator.__sklearn_tags__ = sklearn_tags_poor_score
            estimator.set_params(max_iter=1)
    try:
        yield
    finally:
        if os.environ.get("SYSTEM_PHASENAME", "") == "codecov":
            _kmeans_init_shapelets.__defaults__ = (10000,)


def patch_check_from_module(check, module):
    if hasattr(module, check.func.__name__):
        patched_check = getattr(module, check.func.__name__)
        return partial(patched_check, *check.args, **check.keywords)

    return  check


try:

    # sklearn 1.6 + with new tags
    BASE_EXPECTED_XFAIL_CHECKS = {
        "check_estimators_pickle": "'Pickling is currently NOT tested!'"
    }

    PER_ESTIMATOR_XFAIL_CHECKS = {
        'KernelKMeans': {
            "check_sample_weight_equivalence_on_dense_data": "Currently not supported due to clusters initialization",
            "check_sample_weight_equivalence_on_sparse_data": "Currently not supported due to clusters initialization"
        },
        'TimeSeriesSVC': {
            "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            "check_sample_weight_equivalence_on_dense_data": "zero sample_weight is not equivalent to removing samples",
            "check_sample_weight_equivalence_on_sparse_data": "zero sample_weight is not equivalent to removing samples"
        },
        'TimeSeriesSVR': {
            "check_fit_idempotent": "Non deterministic",
            "check_sample_weights_invariance": "zero sample_weight is not equivalent to removing samples",
            "check_sample_weight_equivalence_on_dense_data": "zero sample_weight is not equivalent to removing samples",
            "check_sample_weight_equivalence_on_sparse_data": "zero sample_weight is not equivalent to removing samples"
        },
        #'PiecewiseAggregateApproximation': {"check_transformer_preserve_dtypes": "Forces int transform"},
        'SymbolicAggregateApproximation': {"check_transformer_preserve_dtypes": "Forces int transform"},
        'OneD_SymbolicAggregateApproximation': {"check_transformer_preserve_dtypes": "Forces int transform"},
        'TimeSeriesImputer': {"check_transformer_data_not_an_array": "Uses X"}
    }

    def get_expected_fails(estimator):
        from sklearn.utils import get_tags

        # Compute expected fails for a given estimator
        expected_fails = PER_ESTIMATOR_XFAIL_CHECKS.get(estimator.__class__.__name__, {})
        expected_fails.update(BASE_EXPECTED_XFAIL_CHECKS)

        return expected_fails

    @parametrize_with_checks(
        get_estimators('all'),
        expected_failed_checks=get_expected_fails
    )
    def test_all_estimators(estimator, check):
        from tslearn.tests import sklearn_patches_new_tags
        actual_check = patch_check_from_module(check, sklearn_patches_new_tags)
        with _configure(estimator, actual_check):
            actual_check(estimator)

except TypeError:
    # sklearn < 1.6, parametrize has only one parameter and uses old tags
    @parametrize_with_checks(get_estimators('all'))
    def test_all_estimators(estimator, check):
        from tslearn.tests import sklearn_patches_old_tags
        actual_check = patch_check_from_module(check, sklearn_patches_old_tags)
        with _configure(estimator, actual_check):
            actual_check(estimator)
