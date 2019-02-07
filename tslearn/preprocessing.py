"""
The :mod:`tslearn.preprocessing` module gathers time series scalers.
"""

import numpy
from sklearn.base import TransformerMixin
from scipy.interpolate import interp1d

from tslearn.utils import to_time_series_dataset, check_equal_size, ts_size

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesResampler(TransformerMixin):
    """Resampler for time series. Resample time series so that they reach the target size.

    Parameters
    ----------
    sz : int
        Size of the output time series.

    Example
    -------
    >>> TimeSeriesResampler(sz=5).fit_transform([[0, 3, 6]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0. ],
            [ 1.5],
            [ 3. ],
            [ 4.5],
            [ 6. ]]])
    """
    def __init__(self, sz):
        self.sz_ = sz

    def fit_transform(self, X, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like
            Time series dataset to be resampled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        X_ = to_time_series_dataset(X)
        n_ts, sz, d = X_.shape
        equal_size = check_equal_size(X_)
        X_out = numpy.empty((n_ts, self.sz_, d))
        for i in range(X_.shape[0]):
            xnew = numpy.linspace(0, 1, self.sz_)
            if not equal_size:
                sz = ts_size(X_[i])
            for di in range(d):
                f = interp1d(numpy.linspace(0, 1, sz), X_[i, :sz, di], kind="slinear")
                X_out[i, :, di] = f(xnew)
        return X_out


class TimeSeriesScalerMinMax(TransformerMixin):
    """Scaler for time series. Scales time series so that their span in each dimension is between ``min`` and ``max``.

    Parameters
    ----------
    min : float (default: 0.)
        Minimum value for output time series.
    max : float (default: 1.)
        Maximum value for output time series.

    Note
    ----
        This method requires a dataset of equal-sized time series.

    Example
    -------
    >>> TimeSeriesScalerMinMax(min=1., max=2.).fit_transform([[0, 3, 6]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1. ],
            [ 1.5],
            [ 2. ]]])
    """
    def __init__(self, min=0., max=1.):
        self.min_ = min
        self.max_ = max

    def fit_transform(self, X, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like
            Time series dataset to be rescaled.

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset.
        """
        X_ = to_time_series_dataset(X)
        min_t = numpy.min(X_, axis=1)[:, numpy.newaxis, :]
        max_t = numpy.max(X_, axis=1)[:, numpy.newaxis, :]
        range_t = max_t - min_t

        X_ = (X_ - min_t) * (self.max_ - self.min_) / range_t + self.min_
        return X_


class TimeSeriesScalerMeanVariance(TransformerMixin):
    """Scaler for time series. Scales time series so that their mean (resp. standard deviation) in each dimension is
    mu (resp. std).

    Parameters
    ----------
    mu : float (default: 0.)
        Mean of the output time series.
    std : float (default: 1.)
        Standard deviation of the output time series.

    Note
    ----
        This method requires a dataset of equal-sized time series.

    Example
    -------
    >>> TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform([[0, 3, 6]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[-1.22474487],
            [ 0. ],
            [ 1.22474487]]])
    """
    def __init__(self, mu=0., std=1.):
        self.mu_ = mu
        self.std_ = std
        self.global_mean = None
        self.global_std = None

    def fit_transform(self, X, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X
            Time series dataset to be rescaled

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        X_ = to_time_series_dataset(X)
        mean_t = numpy.mean(X_, axis=1)[:, numpy.newaxis, :]
        std_t = numpy.std(X_, axis=1)[:, numpy.newaxis, :]
        std_t[std_t == 0.] = 1.

        X_ = (X_ - mean_t) * self.std_ / std_t + self.mu_

        return X_
