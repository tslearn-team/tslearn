"""
The :mod:`tslearn.matrix_profile` module gathers methods for the computation of
Matrix Profiles from time series.
"""

import numpy
from scipy.spatial.distance import pdist, squareform
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array

from tslearn.utils import check_dims
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.bases import BaseModelPackage, TimeSeriesBaseEstimator


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class MatrixProfile(TransformerMixin,
                    BaseModelPackage,
                    TimeSeriesBaseEstimator):
    """TODO
    """

    def __init__(self, subsequence_legth=1):
        self.subsequence_length = subsequence_legth

    def _is_fitted(self):
        return True

    def fit(self, X, y=None):
        """TODO
        """
        return self

    def _transform(self, X, y=None):
        n_ts, sz, d = X.shape
        output_dim = sz - self.subsequence_length + 1
        X_transformed = numpy.empty((n_ts, output_dim))
        scaler = TimeSeriesScalerMeanVariance()
        for i_ts in range(n_ts):
            segments = numpy.array(
                [X[i_ts, t:t+self.subsequence_length]
                 for t in range(output_dim)]
            )  # TODO: look for a pure numpy way to do this
            segments = scaler.fit_transform(segments)
            dists = squareform(pdist(segments, "euclidean"))
            numpy.fill_diagonal(dists, numpy.inf)
            X_transformed[i_ts] = dists.min(axis=0)
        return X_transformed

    def transform(self, X, y=None):
        """TODO

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, output_dim)
            PAA-Transformed dataset
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True)
        X = check_dims(X)
        return self._transform(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """TODO

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, n_segments, d)
            PAA-Transformed dataset
        """
        X = check_array(X, allow_nd=True)
        X = check_dims(X)
        return self.fit(X)._transform(X)
