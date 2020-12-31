import numpy
from scipy.stats import norm
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from tslearn.bases import BaseModelPackage, TimeSeriesBaseEstimator
from tslearn.metrics.cysax import (cydist_sax, cyslopes, cydist_1d_sax,
                                   inv_transform_1d_sax, inv_transform_sax,
                                   inv_transform_paa)
from tslearn.utils import ts_size, check_dims

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _paa_to_symbols(X_paa, breakpoints):
    """Transforms a Piecewise Aggregate Approximation representation into a
    SAX one given breakpoints.

    Examples
    --------
    >>> _paa_to_symbols(X_paa=numpy.array([-1., 0.1, 2.]),
    ...                 breakpoints=numpy.array([0.]))
    array([0, 1, 1])
    """
    alphabet_size = breakpoints.shape[0] + 1
    X_symbols = numpy.zeros(X_paa.shape, dtype=numpy.int) - 1
    for idx_bp, bp in enumerate(breakpoints):
        indices = numpy.logical_and(X_symbols < 0, X_paa < bp)
        X_symbols[indices] = idx_bp
    X_symbols[X_symbols < 0] = alphabet_size - 1
    return X_symbols


def _breakpoints(n_bins, scale=1.):
    """Compute breakpoints for a given number of SAX symbols and a given
    Gaussian scale.

    Examples
    --------
    >>> _breakpoints(n_bins=2)
    array([0.])
    """
    return norm.ppf([float(a) / n_bins for a in range(1, n_bins)], scale=scale)


def _bin_medians(n_bins, scale=1.):
    """Compute median value corresponding to SAX symbols for a given Gaussian
    scale.

    Examples
    --------
    >>> _bin_medians(n_bins=2)
    array([-0.67448975,  0.67448975])
    """
    return norm.ppf([float(a) / (2 * n_bins) for a in range(1, 2 * n_bins, 2)],
                    scale=scale)


class PiecewiseAggregateApproximation(TransformerMixin,
                                      BaseModelPackage,
                                      TimeSeriesBaseEstimator):
    """Piecewise Aggregate Approximation (PAA) transformation.

    PAA was originally presented in [1]_.

    Parameters
    ----------
    n_segments : int (default: 1)
        Number of PAA segments to compute

    Notes
    -----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> paa = PiecewiseAggregateApproximation(n_segments=3)
    >>> data = [[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]]
    >>> paa_data = paa.fit_transform(data)
    >>> paa_data.shape
    (2, 3, 1)
    >>> paa_data
    array([[[ 0.5 ],
            [-0.45],
            [ 0.  ]],
    <BLANKLINE>
           [[ 2.1 ],
            [-2.  ],
            [ 0.  ]]])
    >>> paa.distance_paa(paa_data[0], paa_data[1])  # doctest: +ELLIPSIS
    3.15039...
    >>> paa.distance(data[0], data[1])  # doctest: +ELLIPSIS
    3.15039...
    >>> paa.inverse_transform(paa_data)
    array([[[ 0.5 ],
            [ 0.5 ],
            [-0.45],
            [-0.45],
            [ 0.  ],
            [ 0.  ]],
    <BLANKLINE>
           [[ 2.1 ],
            [ 2.1 ],
            [-2.  ],
            [-2.  ],
            [ 0.  ],
            [ 0.  ]]])

    References
    ----------
    .. [1] E. Keogh & M. Pazzani. Scaling up dynamic time warping for
       datamining applications. SIGKDD 2000, pp. 285--289.
    """

    def __init__(self, n_segments=1):
        self.n_segments = n_segments

    def _is_fitted(self):
        check_is_fitted(self, '_X_fit_dims_')
        return True

    def _fit(self, X, y=None):
        self._X_fit_dims_ = numpy.array(X.shape)
        return self

    def fit(self, X, y=None):
        """Fit a PAA representation.

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
        return self._fit(X, y)

    def _transform(self, X, y=None):
        n_ts, sz, d = X.shape
        X_transformed = numpy.empty((n_ts, self.n_segments, d))
        for i_ts in range(n_ts):
            sz_segment = ts_size(X[i_ts]) // self.n_segments
            for i_seg in range(self.n_segments):
                start = i_seg * sz_segment
                end = start + sz_segment
                segment = X[i_ts, start:end, :]
                X_transformed[i_ts, i_seg, :] = segment.mean(axis=0)
        return X_transformed

    def transform(self, X, y=None):
        """Transform a dataset of time series into its PAA representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, n_segments, d)
            PAA-Transformed dataset
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X, X_fit_dims=tuple(self._X_fit_dims_),
                       check_n_features_only=True)
        return self._transform(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit a PAA representation and transform the data accordingly.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, n_segments, d)
            PAA-Transformed dataset
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)._transform(X)

    def distance_paa(self, paa1, paa2):
        """Compute distance between PAA representations as defined in [1]_.

        Parameters
        ----------
        paa1 : array-like
            PAA representation of a time series
        paa2 : array-like
            PAA representation of another time series

        Returns
        -------
        float
            PAA distance

        References
        ----------
        .. [1] E. Keogh & M. Pazzani. Scaling up dynamic time warping for
           datamining applications. SIGKDD 2000, pp. 285--289.
        """
        self._is_fitted()
        return (numpy.linalg.norm(paa1 - paa2) *
                numpy.sqrt(self._X_fit_dims_[1] / self.n_segments))

    def distance(self, ts1, ts2):
        """Compute distance between PAA representations as defined in [1]_.

        Parameters
        ----------
        ts1 : array-like
            A time series
        ts2 : array-like
            Another time series

        Returns
        -------
        float
            PAA distance

        References
        ----------
        .. [1] E. Keogh & M. Pazzani. Scaling up dynamic time warping for
           datamining applications. SIGKDD 2000, pp. 285--289.
        """
        paa = self.transform([ts1, ts2])
        return self.distance_paa(paa[0], paa[1])

    def inverse_transform(self, X):
        """Compute time series corresponding to given PAA representations.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz_paa, d)
            A dataset of PAA series.

        Returns
        -------
        numpy.ndarray of shape (n_ts, sz_original_ts, d)
            A dataset of time series corresponding to the provided
            representation.
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return inv_transform_paa(X, original_size=self._X_fit_dims_[1])

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}


class SymbolicAggregateApproximation(PiecewiseAggregateApproximation):
    """Symbolic Aggregate approXimation (SAX) transformation.

    SAX was originally presented in [1]_.

    Parameters
    ----------
    n_segments : int (default: 1)
        Number of PAA segments to compute
        
    alphabet_size_avg : int (default: 5)
        Number of SAX symbols to use
    
    scale: bool (default: False)
         Whether input data should be scaled for each feature to have zero 
         mean and unit variance across the dataset passed at fit time.
         Default for this parameter is set to `False` in version 0.4 to ensure
         backward compatibility, but is likely to change in a future version.

    Attributes
    ----------
    breakpoints_avg_ : numpy.ndarray of shape (alphabet_size - 1, )
        List of breakpoints used to generate SAX symbols

    Notes
    -----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> sax = SymbolicAggregateApproximation(n_segments=3, alphabet_size_avg=2)
    >>> data = [[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]]
    >>> sax_data = sax.fit_transform(data)
    >>> sax_data.shape
    (2, 3, 1)
    >>> sax_data
    array([[[1],
            [0],
            [1]],
    <BLANKLINE>
           [[1],
            [0],
            [1]]])
    >>> sax.distance_sax(sax_data[0], sax_data[1])  # doctest: +ELLIPSIS
    0.0
    >>> sax.distance(data[0], data[1])  # doctest: +ELLIPSIS
    0.0
    >>> sax.inverse_transform(sax_data)
    array([[[ 0.67448975],
            [ 0.67448975],
            [-0.67448975],
            [-0.67448975],
            [ 0.67448975],
            [ 0.67448975]],
    <BLANKLINE>
           [[ 0.67448975],
            [ 0.67448975],
            [-0.67448975],
            [-0.67448975],
            [ 0.67448975],
            [ 0.67448975]]])

    References
    ----------
    .. [1] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel symbolic
       representation of time series. Data Mining and Knowledge Discovery,
       2007. vol. 15(107)
    """
    def __init__(self, n_segments=1, alphabet_size_avg=5, scale=False):
        super().__init__(n_segments=n_segments)
        self.alphabet_size_avg = alphabet_size_avg
        self.scale = scale

    def _is_fitted(self):
        check_is_fitted(self, ['breakpoints_avg_', 'breakpoints_avg_middle_'])
        if self.scale:
            check_is_fitted(self, ['mu_', 'std_'])
        return super()._is_fitted()

    def _fit(self, X, y=None):
        self.breakpoints_avg_ = _breakpoints(self.alphabet_size_avg)
        self.breakpoints_avg_middle_ = _bin_medians(self.alphabet_size_avg)

        if self.scale:
            d = X.shape[2]
            reshaped_X = X.reshape((-1, d))
            self.mu_ = numpy.nanmean(reshaped_X, axis=0)
            self.std_ = numpy.nanstd(reshaped_X, axis=0)
            self.std_[self.std_ == 0.] = 1.

        return super()._fit(X, y)

    def fit(self, X, y=None):
        """Fit a SAX representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        SymbolicAggregateApproximation
            self
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit a SAX representation and transform the data accordingly.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of integers with shape (n_ts, n_segments, d)
            SAX-Transformed dataset
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)._transform(X)

    def _scale(self, X):
        if not self.scale:
            return X

        std = self.std_.reshape((1, 1, -1))
        mu = self.mu_.reshape((1, 1, -1))

        return (X - mu) / std

    def _unscale(self, X):
        if not self.scale:
            return X

        std = self.std_.reshape((1, 1, -1))
        mu = self.mu_.reshape((1, 1, -1))

        return X * std + mu

    def _transform(self, X, y=None):
        X = self._scale(X)
        X_paa = super()._transform(X, y)
        return _paa_to_symbols(X_paa, self.breakpoints_avg_)

    def transform(self, X, y=None):
        """Transform a dataset of time series into its SAX representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of integers with shape (n_ts, n_segments, d)
            SAX-Transformed dataset
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X, X_fit_dims=tuple(self._X_fit_dims_),
                       check_n_features_only=True)
        return self._transform(X, y)

    def distance_sax(self, sax1, sax2):
        """Compute distance between SAX representations as defined in [1]_.

        Parameters
        ----------
        sax1 : array-like
            SAX representation of a time series
        sax2 : array-like
            SAX representation of another time series

        Returns
        -------
        float
            SAX distance

        References
        ----------
        .. [1] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel
           symbolic representation of time series.
           Data Mining and Knowledge Discovery, 2007. vol. 15(107)
        """
        self._is_fitted()
        return cydist_sax(sax1, sax2,
                          self.breakpoints_avg_, self._X_fit_dims_[1])

    def distance(self, ts1, ts2):
        """Compute distance between SAX representations as defined in [1]_.

        Parameters
        ----------
        ts1 : array-like
            A time series
        ts2 : array-like
            Another time series

        Returns
        -------
        float
            SAX distance

        References
        ----------
        .. [1] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel
           symbolic representation of time series. Data Mining and Knowledge
           Discovery, 2007. vol. 15(107)
        """
        sax = self.transform([ts1, ts2])
        return self.distance_sax(sax[0], sax[1])

    def inverse_transform(self, X):
        """Compute time series corresponding to given SAX representations.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz_sax, d)
            A dataset of SAX series.

        Returns
        -------
        numpy.ndarray of shape (n_ts, sz_original_ts, d)
            A dataset of time series corresponding to the provided
            representation.
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X, X_fit_dims=(None, None, self._X_fit_dims_[-1]),
                       check_n_features_only=True)
        X_orig = inv_transform_sax(
                X,
                breakpoints_middle_=self.breakpoints_avg_middle_,
                original_size=self._X_fit_dims_[1]
        )
        return self._unscale(X_orig)


class OneD_SymbolicAggregateApproximation(SymbolicAggregateApproximation):
    """One-D Symbolic Aggregate approXimation (1d-SAX) transformation.

    1d-SAX was originally presented in [1]_.

    Parameters
    ----------
    n_segments : int (default: 1)
        Number of PAA segments to compute.
        
    alphabet_size_avg : int (default: 5)
        Number of SAX symbols to use to describe average values.
        
    alphabet_size_slope : int (default: 5)
        Number of SAX symbols to use to describe slopes.
        
    sigma_l : float or None (default: None)
        Scale parameter of the Gaussian distribution used to quantize slopes.
        If None, the formula given in [1]_ is
        used: :math:`\\sigma_L = \\sqrt{0.03 / L}` where :math:`L` is the
        length of each segment.
    
    scale: bool (default: False)
         Whether input data should be scaled for each feature of each time 
         series to have zero mean and unit variance.
         Default for this parameter is set to `False` in version 0.4 to ensure
         backward compatibility, but is likely to change in a future version.

    Attributes
    ----------
    breakpoints_avg_ : numpy.ndarray of shape (alphabet_size_avg - 1, )
        List of breakpoints used to generate SAX symbols for average values.
    breakpoints_slope_ : numpy.ndarray of shape (alphabet_size_slope - 1, )
        List of breakpoints used to generate SAX symbols for slopes.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=3,
    ...         alphabet_size_avg=2, alphabet_size_slope=2, sigma_l=1.)
    >>> data = [[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]]
    >>> one_d_sax_data = one_d_sax.fit_transform(data)
    >>> one_d_sax_data.shape
    (2, 3, 2)
    >>> one_d_sax_data
    array([[[1, 1],
            [0, 0],
            [1, 0]],
    <BLANKLINE>
           [[1, 1],
            [0, 0],
            [1, 0]]])
    >>> one_d_sax.distance_sax(one_d_sax_data[0], one_d_sax_data[1])
    0.0
    >>> one_d_sax.distance(data[0], data[1])
    0.0
    >>> one_d_sax.inverse_transform(one_d_sax_data)
    array([[[ 0.33724488],
            [ 1.01173463],
            [-0.33724488],
            [-1.01173463],
            [ 1.01173463],
            [ 0.33724488]],
    <BLANKLINE>
           [[ 0.33724488],
            [ 1.01173463],
            [-0.33724488],
            [-1.01173463],
            [ 1.01173463],
            [ 0.33724488]]])
    >>> one_d_sax.fit(data).sigma_l
    1.0


    References
    ----------
    .. [1] S. Malinowski, T. Guyet, R. Quiniou, R. Tavenard. 1d-SAX: a Novel
       Symbolic Representation for Time Series. IDA 2013.
    """
    def __init__(self, n_segments=1, alphabet_size_avg=5,
                 alphabet_size_slope=5, sigma_l=None, scale=False):
        super().__init__(
            n_segments=n_segments,
            alphabet_size_avg=alphabet_size_avg,
            scale=scale
        )
        self.alphabet_size_slope = alphabet_size_slope
        self.sigma_l = sigma_l

    def _is_fitted(self):
        check_is_fitted(self,
                        ['breakpoints_slope_', 'breakpoints_slope_middle_'])
        return super()._is_fitted()

    def _fit(self, X, y=None):
        super()._fit(X, y)

        n_ts, sz, d = X.shape
        sz_segment = sz // self.n_segments
        sigma_l = self.sigma_l
        if sigma_l is None:
            sigma_l = numpy.sqrt(0.03 / sz_segment)

        # Do that at fit time when we have sigma_l for sure
        self.breakpoints_slope_ = _breakpoints(self.alphabet_size_slope,
                                               scale=sigma_l)
        self.breakpoints_slope_middle_ = _bin_medians(self.alphabet_size_slope,
                                                      scale=sigma_l)
        return self

    def fit(self, X, y=None):
        """Fit a 1d-SAX representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        OneD_SymbolicAggregateApproximation
            self
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit a 1d-SAX representation and transform the data accordingly.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of integers with shape (n_ts, n_segments, 2 * d)
            1d-SAX-Transformed dataset. The order of the last dimension is:
            first d elements represent average values
            (standard SAX symbols) and the last d are for slopes
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)._transform(X)

    def _get_slopes(self, X):
        n_ts, sz, d = X.shape
        X_slopes = numpy.empty((n_ts, self.n_segments, d))
        sz_segment = sz // self.n_segments
        for i_seg in range(self.n_segments):
            start = i_seg * sz_segment
            end = start + sz_segment
            X_slopes[:, i_seg, :] = cyslopes(X[:, start:end, :], start)
        return X_slopes

    def _transform(self, X, y=None):
        X = self._scale(X)
        n_ts, sz_raw, d = X.shape
        X_1d_sax = numpy.empty((n_ts, self.n_segments, 2 * d), dtype=numpy.int)

        # Average
        X_1d_sax_avg = SymbolicAggregateApproximation._transform(self, X)

        # Slope
        X_slopes = self._get_slopes(X)
        X_1d_sax_slope = _paa_to_symbols(X_slopes, self.breakpoints_slope_)

        X_1d_sax[:, :, :d] = X_1d_sax_avg
        X_1d_sax[:, :, d:] = X_1d_sax_slope

        return X_1d_sax

    def transform(self, X, y=None):
        """Transform a dataset of time series into its 1d-SAX representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of integers with shape (n_ts, n_segments, 2 * d)
            1d-SAX-Transformed dataset
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True, dtype=numpy.float,
                        force_all_finite=False)
        X = check_dims(X, X_fit_dims=tuple(self._X_fit_dims_),
                       check_n_features_only=True)
        return self._transform(X, y)

    def distance_1d_sax(self, sax1, sax2):
        """Compute distance between 1d-SAX representations as defined in [1]_.

        Parameters
        ----------
        sax1 : array-like
            1d-SAX representation of a time series
        sax2 : array-like
            1d-SAX representation of another time series

        Returns
        -------
        float
            1d-SAX distance

        Notes
        -----
            Unlike SAX distance, 1d-SAX distance does not lower bound Euclidean
            distance between original time series.

        References
        ----------
        .. [1] S. Malinowski, T. Guyet, R. Quiniou, R. Tavenard. 1d-SAX: a
           Novel Symbolic Representation for Time Series. IDA 2013.
        """
        self._is_fitted()
        return cydist_1d_sax(sax1, sax2, self.breakpoints_avg_middle_,
                             self.breakpoints_slope_middle_,
                             self._X_fit_dims_[1])

    def distance(self, ts1, ts2):
        """Compute distance between 1d-SAX representations as defined in [1]_.

        Parameters
        ----------
        ts1 : array-like
            A time series
        ts2 : array-like
            Another time series

        Returns
        -------
        float
            1d-SAX distance

        References
        ----------
        .. [1] S. Malinowski, T. Guyet, R. Quiniou, R. Tavenard. 1d-SAX: a
           Novel Symbolic Representation for Time Series. IDA 2013.
        """
        sax1d = self.transform([ts1, ts2])
        return self.distance_1d_sax(sax1d[0], sax1d[1])

    def inverse_transform(self, X):
        """Compute time series corresponding to given 1d-SAX representations.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz_sax, 2 * d)
            A dataset of SAX series.

        Returns
        -------
        numpy.ndarray of shape (n_ts, sz_original_ts, d)
            A dataset of time series corresponding to the provided
            representation.
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True)
        X = check_dims(X, X_fit_dims=(None, None, 2 * self._X_fit_dims_[-1]),
                       check_n_features_only=True)
        X_orig = inv_transform_1d_sax(
                X,
                breakpoints_avg_middle_=self.breakpoints_avg_middle_,
                breakpoints_slope_middle_=self.breakpoints_slope_middle_,
                original_size=self._X_fit_dims_[1]
        )
        return self._unscale(X_orig)
