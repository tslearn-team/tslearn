"""
The :mod:`tslearn.matrix_profile` module gathers methods for the computation of
Matrix Profiles from time series.
"""

import numpy
from numpy.lib.stride_tricks import as_strided
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
    
    Examples
    --------
    >>> time_series = [0., 1., 3., 2., 9., 1., 14., 15., 1., 2., 2., 10., 7.]
    >>> ds = [time_series]
    >>> mp = MatrixProfile(subsequence_length=4, scale=False)
    >>> mp.fit_transform(ds)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 6.85...,  1.41...,  6.16..., 7.93..., 11.40...,
            13.56..., 14.07..., 13.96..., 1.41...,  6.16...]])
    """

    def __init__(self, subsequence_length=1, scale=True):
        self.subsequence_length = subsequence_length
        self.scale = scale

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
            Xi = X[i_ts]
            elem_size = Xi.strides[0]
            segments = as_strided(
                Xi,
                strides=(elem_size, elem_size, Xi.strides[1]),
                shape=(Xi.shape[0] - self.subsequence_length + 1,
                       self.subsequence_length, d),
                writeable=False
            )
            if self.scale:
                segments = scaler.fit_transform(segments)
            segments_2d = segments.reshape((-1, self.subsequence_length * d))
            dists = squareform(pdist(segments_2d, "euclidean"))
            numpy.fill_diagonal(dists, numpy.inf)
            X_transformed[i_ts] = dists.min(axis=1)
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
