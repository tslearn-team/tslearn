"""
The :mod:`tslearn.svm` module contains Support Vector Classifier (SVC) and Support Vector Regressor (SVR) models
for time series.
"""

from __future__ import print_function
from sklearn.svm import SVC as BaseSVC, SVR as BaseSVR
import numpy

from tslearn.metrics import cdist_gak
from tslearn.utils import to_time_series_dataset


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesSVC(BaseSVC):
    """Time-series specific Support Vector Classifier.

    Parameters
    ----------
    TODO

    Attributes
    ----------
    TODO

    Examples
    --------
    TODO

    References
    ----------
    Fast Global Alignment Kernels.
    Marco Cuturi.
    ICML 2011.
    """
    def __init__(self, C=1.0, kernel="gak", sz=None, d=None, degree=3, gamma="auto", coef0=0.0, shrinking=True,
                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape="ovr", random_state=None):  # TODO: gamma auto
        if kernel == "gak":
            kernel = lambda x, y: cdist_gak(x.reshape((-1, sz, d)), y.reshape((-1, sz, d)),
                                            sigma=numpy.sqrt(gamma / 2.))
        super(TimeSeriesSVC, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                                            shrinking=shrinking, probability=probability, tol=tol,
                                            cache_size=cache_size, class_weight=class_weight, verbose=verbose,
                                            max_iter=max_iter, decision_function_shape=decision_function_shape,
                                            random_state=random_state)

    def fit(self, X, y, sample_weight=None):
        sklearn_X = to_time_series_dataset(X)
        n_ts, sz, d = sklearn_X.shape
        sklearn_X = sklearn_X.reshape((n_ts, -1))
        return super(TimeSeriesSVC, self).fit(sklearn_X, y, sample_weight=sample_weight)

    def predict(self, X):
        sklearn_X = to_time_series_dataset(X)
        n_ts, sz, d = sklearn_X.shape
        sklearn_X = sklearn_X.reshape((n_ts, -1))
        return super(TimeSeriesSVC, self).predict(sklearn_X)
