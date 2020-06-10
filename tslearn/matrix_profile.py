"""
The :mod:`tslearn.matrix_profile` module gathers methods for the computation of
Matrix Profiles from time series.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.distance import pdist, squareform
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from tslearn.utils import check_dims
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.bases import BaseModelPackage, TimeSeriesBaseEstimator


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class MatrixProfile(TransformerMixin,
                    BaseModelPackage,
                    TimeSeriesBaseEstimator):
    """Matrix Profile transformation.

    Matrix Profile was originally presented in [1]_.

    Parameters
    ----------
    subsequence_length : int (default: 1)
        Length of the subseries (also called window size) to be used for 
        subseries distance computations.
    
    scale: bool (default: True)
         Whether input data should be scaled for each feature of each time 
         series to have zero mean and unit variance.
         Default for this parameter is set to `True` to match the standard 
         matrix profile setup.
    
    Examples
    --------
    >>> time_series = [0., 1., 3., 2., 9., 1., 14., 15., 1., 2., 2., 10., 7.]
    >>> ds = [time_series]
    >>> mp = MatrixProfile(subsequence_length=4, scale=False)
    >>> mp.fit_transform(ds)[0, :, 0]  # doctest: +ELLIPSIS
    array([ 6.85...,  1.41...,  6.16...,  7.93..., 11.40...,
           13.56..., 18.  ..., 13.96...,  1.41...,  6.16...])

    References
    ----------
    .. [1] C. M. Yeh, Y. Zhu, L. Ulanova, N.Begum et al. 
       Matrix Profile I: All Pairs Similarity Joins for Time Series: A 
       Unifying View that Includes Motifs, Discords and Shapelets.
       ICDM 2016.
    """

    def __init__(self, subsequence_length=1, scale=True):
        self.subsequence_length = subsequence_length
        self.scale = scale

    def _is_fitted(self):
        check_is_fitted(self, '_X_fit_dims')
        return True

    def _fit(self, X, y=None):
        self._X_fit_dims = X.shape
        return self

    def fit(self, X, y=None):
        """Fit a Matrix Profile representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        PiecewiseAggregateApproximation
            self
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)

    def _transform(self, X, y=None):
        n_ts, sz, d = X.shape
        output_size = sz - self.subsequence_length + 1
        X_transformed = np.empty((n_ts, output_size, 1))
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
            n_segments = segments.shape[0]
            segments_2d = segments.reshape((-1, self.subsequence_length * d))
            dists = squareform(pdist(segments_2d, "euclidean"))
            band = (np.tri(n_segments, n_segments,
                           self.subsequence_length / 4, dtype=np.bool) &
                    ~np.tri(n_segments, n_segments,
                            -(self.subsequence_length / 4 + 1), dtype=np.bool))
            dists[band] = np.inf
            X_transformed[i_ts] = dists.min(axis=1, keepdims=True)
        return X_transformed

    def transform(self, X, y=None):
        """Transform a dataset of time series into its Matrix Profile
         representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, output_size, 1)
            Matrix-Profile-Transformed dataset. `ouput_size` is equal to 
            `sz - subsequence_length + 1`
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        return self._transform(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Transform a dataset of time series into its Matrix Profile
         representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, output_size, 1)
            Matrix-Profile-Transformed dataset. `ouput_size` is equal to 
            `sz - subsequence_length + 1`
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)._transform(X)

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}
