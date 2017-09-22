"""
The :mod:`tslearn.piecewise` module gathers time series piecewise approximation algorithms.
"""

import numpy
from scipy.stats import norm
from sklearn.base import TransformerMixin

from tslearn.utils import to_time_series_dataset
from tslearn.cysax import cydist_sax, cyslopes, cydist_1d_sax, inv_transform_1d_sax, inv_transform_sax, inv_transform_paa


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _paa_to_symbols(X_paa, breakpoints):
    """Transforms a Piecewise Aggregate Approximation representation into a SAX one given breakpoints.

    Example
    -------
    >>> _paa_to_symbols(X_paa=numpy.array([-1., 0.1, 2.]), breakpoints=numpy.array([0.]))
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
    """Compute breakpoints for a given number of SAX symbols and a given Gaussian scale.

    Example
    -------
    >>> _breakpoints(n_bins=2)
    array([ 0.])
    """
    return norm.ppf([float(a) / n_bins for a in range(1, n_bins)], scale=scale)


def _bin_medians(n_bins, scale=1.):
    """Compute median value corresponding to SAX symbols for a given Gaussian scale.

    Example
    -------
    >>> _bin_medians(n_bins=2)
    array([-0.67448975,  0.67448975])
    """
    return norm.ppf([float(a) / (2 * n_bins) for a in range(1, 2 * n_bins, 2)], scale=scale)


class PiecewiseAggregateApproximation(TransformerMixin):
    """Piecewise Aggregate Approximation (PAA) transformation.

    PAA was originally presented in [1]_.

    Parameters
    ----------
    n_segments : int
        Number of PAA segments to compute

    Note
    ----
        This method requires a dataset of equal-sized time series.

    Example
    -------
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
    >>> numpy.alltrue(paa_data == paa.fit(data).transform(data))
    True
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
    >>> unfitted_paa = PiecewiseAggregateApproximation(n_segments=3)
    >>> unfitted_paa.distance(data[0], data[1])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Model not fitted yet: cannot be used for distance computation.

    References
    ----------
    .. [1] E. Keogh & M. Pazzani. Scaling up dynamic time warping for datamining applications. SIGKDD 2000,
       pp. 285--289.
    """
    def __init__(self, n_segments):
        self.n_segments = n_segments
        self.size_fitted_ = -1

    def _fit(self, X, y=None):
        self.size_fitted_ = X.shape[1]
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
        X_ = to_time_series_dataset(X)
        return self._fit(X_, y)

    def _transform(self, X, y=None):
        n_ts, sz, d = X.shape
        X_transformed = numpy.empty((n_ts, self.n_segments, d))
        sz_segment = sz // self.n_segments
        for i_seg in range(self.n_segments):
            start = i_seg * sz_segment
            end = start + sz_segment
            X_transformed[:, i_seg, :] = X[:, start:end, :].mean(axis=1)
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
        X_ = to_time_series_dataset(X)
        return self._transform(X_, y)

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
        X_ = to_time_series_dataset(X)
        return self._fit(X_)._transform(X_)

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
        .. [1] E. Keogh & M. Pazzani. Scaling up dynamic time warping for datamining applications. SIGKDD 2000,
           pp. 285--289.
        """
        if self.size_fitted_ < 0:
            raise ValueError("Model not fitted yet: cannot be used for distance computation.")
        else:
            return numpy.linalg.norm(paa1 - paa2) * numpy.sqrt(self.size_fitted_ / self.n_segments)

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
        .. [1] E. Keogh & M. Pazzani. Scaling up dynamic time warping for datamining applications. SIGKDD 2000,
           pp. 285--289.
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
            A dataset of time series corresponding to the provided representation.
        """
        X_ = to_time_series_dataset(X)
        return inv_transform_paa(X_, original_size=self.size_fitted_)


class SymbolicAggregateApproximation(PiecewiseAggregateApproximation):
    """Symbolic Aggregate approXimation (SAX) transformation.

    SAX was originally presented in [1]_.

    Parameters
    ----------
    n_segments : int
        Number of PAA segments to compute
    alphabet_size_avg : int
        Number of SAX symbols to use

    Attributes
    ----------
    breakpoints_avg_ : numpy.ndarray of shape (alphabet_size - 1, )
        List of breakpoints used to generate SAX symbols

    Note
    ----
        This method requires a dataset of equal-sized time series.

    Example
    -------
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
    >>> numpy.alltrue(sax_data == sax.fit(data).transform(data))
    True
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
    >>> unfitted_sax = SymbolicAggregateApproximation(n_segments=3, alphabet_size_avg=2)
    >>> unfitted_sax.distance(data[0], data[1])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Model not fitted yet: cannot be used for distance computation.

    References
    ----------
    .. [1] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel symbolic representation of time series.
       Data Mining and Knowledge Discovery, 2007. vol. 15(107)
    """
    def __init__(self, n_segments, alphabet_size_avg):
        PiecewiseAggregateApproximation.__init__(self, n_segments)
        self.alphabet_size_avg = alphabet_size_avg
        self.breakpoints_avg_ = _breakpoints(self.alphabet_size_avg)
        self.breakpoints_avg_middle_ = _bin_medians(self.alphabet_size_avg)

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
        return PiecewiseAggregateApproximation.fit(self, X, y)

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
        X_ = to_time_series_dataset(X)
        return self._fit(X_)._transform(X_)

    def _transform(self, X, y=None):
        X_paa = PiecewiseAggregateApproximation._transform(self, X, y)
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
        X_ = to_time_series_dataset(X)
        return self._transform(X_, y)

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
        .. [1] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel symbolic representation of time series.
           Data Mining and Knowledge Discovery, 2007. vol. 15(107)
        """
        if self.size_fitted_ < 0:
            raise ValueError("Model not fitted yet: cannot be used for distance computation.")
        else:
            return cydist_sax(sax1, sax2, self.breakpoints_avg_, self.size_fitted_)

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
        .. [1] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel symbolic representation of time series.
           Data Mining and Knowledge Discovery, 2007. vol. 15(107)
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
            A dataset of time series corresponding to the provided representation.
        """
        X_ = numpy.array(X, dtype=numpy.int)
        return inv_transform_sax(X_, breakpoints_middle_=self.breakpoints_avg_middle_, original_size=self.size_fitted_)


class OneD_SymbolicAggregateApproximation(SymbolicAggregateApproximation):
    """One-D Symbolic Aggregate approXimation (1d-SAX) transformation.

    1d-SAX was originally presented in [1]_.

    Parameters
    ----------
    n_segments : int
        Number of PAA segments to compute.
    alphabet_size_avg : int
        Number of SAX symbols to use to describe average values.
    alphabet_size_slope : int
        Number of SAX symbols to use to describe slopes.
    sigma_l : float or None (default: None)
        Scale parameter of the Gaussian distribution used to quantize slopes. If None, the formula given in [1]_ is
        used: :math:`\\sigma_L = \\sqrt{0.03 / L}` where :math:`L` is the length of the considered time series.

    Attributes
    ----------
    breakpoints_avg_ : numpy.ndarray of shape (alphabet_size_avg - 1, )
        List of breakpoints used to generate SAX symbols for average values.
    breakpoints_slope_ : numpy.ndarray of shape (alphabet_size_slope - 1, )
        List of breakpoints used to generate SAX symbols for slopes.

    Note
    ----
        This method requires a dataset of equal-sized time series.

    Example
    -------
    >>> one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=3, alphabet_size_avg=2, alphabet_size_slope=2, sigma_l=1.)
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
    >>> numpy.alltrue(one_d_sax_data == one_d_sax.fit(data).transform(data))
    True
    >>> one_d_sax.distance_sax(one_d_sax_data[0], one_d_sax_data[1])  # doctest: +ELLIPSIS
    0.0
    >>> one_d_sax.distance(data[0], data[1])  # doctest: +ELLIPSIS
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
    >>> unfitted_1dsax = OneD_SymbolicAggregateApproximation(n_segments=3, alphabet_size_avg=2, alphabet_size_slope=2)
    >>> unfitted_1dsax.distance(data[0], data[1])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Model not fitted yet: cannot be used for distance computation.
    >>> unfitted_1dsax.fit(data).sigma_l  # doctest: +ELLIPSIS
    0.07071...


    References
    ----------
    .. [1] S. Malinowski, T. Guyet, R. Quiniou, R. Tavenard. 1d-SAX: a Novel Symbolic Representation for Time Series.
       IDA 2013.
    """
    def __init__(self, n_segments, alphabet_size_avg, alphabet_size_slope, sigma_l=None):
        SymbolicAggregateApproximation.__init__(self, n_segments, alphabet_size_avg=alphabet_size_avg)
        self.alphabet_size_slope = alphabet_size_slope
        self.sigma_l = sigma_l

        self.breakpoints_slope_ = None  # Do that at fit time when we have sigma_l for sure
        self.breakpoints_slope_middle_ = None

    def _fit(self, X, y=None):
        SymbolicAggregateApproximation._fit(self, X, y)
        if self.sigma_l is None:
            self.sigma_l = numpy.sqrt(0.03 / self.size_fitted_)

        self.breakpoints_slope_ = _breakpoints(self.alphabet_size_slope, scale=self.sigma_l)
        self.breakpoints_slope_middle_ = _bin_medians(self.alphabet_size_slope, scale=self.sigma_l)
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
        X_ = to_time_series_dataset(X)
        return self._fit(X_)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit a 1d-SAX representation and transform the data accordingly.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of integers with shape (n_ts, n_segments, 2 * d)
            1d-SAX-Transformed dataset. The order of the last dimension is: first d elements represent average values
            (standard SAX symbols) and the last d are for slopes
        """
        X_ = to_time_series_dataset(X)
        return self._fit(X_)._transform(X_)

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
        if self.size_fitted_ < 0:
            raise ValueError("Model not fitted yet: cannot be used for distance computation.")
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
        X_ = to_time_series_dataset(X)
        return self._transform(X_, y)

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

        Note
        ----
            Unlike SAX distance, 1d-SAX distance does not lower bound Euclidean distance between original time series.

        References
        ----------
        .. [1] S. Malinowski, T. Guyet, R. Quiniou, R. Tavenard. 1d-SAX: a Novel Symbolic Representation for Time
           Series. IDA 2013.
        """
        if self.size_fitted_ < 0:
            raise ValueError("Model not fitted yet: cannot be used for distance computation.")
        else:
            return cydist_1d_sax(sax1, sax2, self.breakpoints_avg_middle_, self.breakpoints_slope_middle_,
                                 self.size_fitted_)

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
        .. [1] S. Malinowski, T. Guyet, R. Quiniou, R. Tavenard. 1d-SAX: a Novel Symbolic Representation for Time
           Series. IDA 2013.
        """
        sax = self.transform([ts1, ts2])
        return self.distance_1d_sax(sax[0], sax[1])

    def inverse_transform(self, X):
        """Compute time series corresponding to given 1d-SAX representations.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz_sax, 2 * d)
            A dataset of SAX series.

        Returns
        -------
        numpy.ndarray of shape (n_ts, sz_original_ts, d)
            A dataset of time series corresponding to the provided representation.
        """
        X_ = numpy.array(X, dtype=numpy.int)
        return inv_transform_1d_sax(X_,
                                    breakpoints_avg_middle_=self.breakpoints_avg_middle_,
                                    breakpoints_slope_middle_=self.breakpoints_slope_middle_,
                                    original_size=self.size_fitted_)
