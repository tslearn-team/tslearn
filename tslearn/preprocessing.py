"""
The :mod:`tslearn.preprocessing` module gathers time series scalers.
"""

import numpy
from sklearn.base import TransformerMixin
from scipy.interpolate import interp1d
import warnings

from tslearn.utils import to_time_series_dataset, check_equal_size, ts_size

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesResampler(TransformerMixin):
    """Resampler for time series. Resample time series so that they reach the
    target size.

    Parameters
    ----------
    sz : int
        Size of the output time series.

    Examples
    --------
    >>> TimeSeriesResampler(sz=5).fit_transform([[0, 3, 6]])
    array([[[0. ],
            [1.5],
            [3. ],
            [4.5],
            [6. ]]])
    """
    def __init__(self, sz):
        self.sz_ = sz

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.

        Parameters
        ----------
        X
            Ignored

        Returns
        -------
        self
        """
        return self

    def transform(self, X, **kwargs):
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
                f = interp1d(numpy.linspace(0, 1, sz), X_[i, :sz, di],
                             kind="slinear")
                X_out[i, :, di] = f(xnew)
        return X_out


class TimeSeriesScalerMinMax(TransformerMixin):
    """Scaler for time series. Scales time series so that their span in each
    dimension is between ``min`` and ``max``.

    Parameters
    ----------
    value_range : tuple (default: (0., 1.))
        The minimum and maximum value for the output time series.

    min : float (default: 0.)
        Minimum value for output time series.

        .. deprecated:: 0.2
            min is deprecated in version 0.2 and will be
            removed in 0.4. Use value_range instead.

    max : float (default: 1.)
        Maximum value for output time series.

        .. deprecated:: 0.2
            min is deprecated in version 0.2 and will be
            removed in 0.4. Use value_range instead.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

        NaNs within a time series are ignored when calculating min and max.

    Examples
    --------
    >>> TimeSeriesScalerMinMax(value_range=(1., 2.)).fit_transform([[0, 3, 6]])
    array([[[1. ],
            [1.5],
            [2. ]]])
    >>> TimeSeriesScalerMinMax(value_range=(1., 2.)).fit_transform(
    ...     [[numpy.nan, 3, 6]]
    ... )
    array([[[nan],
            [ 1.],
            [ 2.]]])
    """
    def __init__(self, value_range=(0., 1.), min=None, max=None):
        self.value_range = value_range
        self.min_ = min
        self.max_ = max

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.

        Parameters
        ----------
        X
            Ignored

        Returns
        -------
        self
        """
        return self

    def transform(self, X, y=None, **kwargs):
        """Will normalize (min-max) each of the timeseries. IMPORTANT: this
        transformation is completely stateless, and is applied to each of
        the timeseries individually.

        Parameters
        ----------
        X : array-like
            Time series dataset to be rescaled.

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset.
        """
        if self.min_ is not None:
            warnings.warn(
                "'min' is deprecated in version 0.2 and will be "
                "removed in 0.4. Use value_range instead.",
                DeprecationWarning, stacklevel=2)
            self.value_range = (self.min_, self.value_range[1])

        if self.max_ is not None:
            warnings.warn(
                "'max' is deprecated in version 0.2 and will be "
                "removed in 0.4. Use value_range instead.",
                DeprecationWarning, stacklevel=2)
            self.value_range = (self.value_range[0], self.max_)

        if self.value_range[0] >= self.value_range[1]:
            raise ValueError("Minimum of desired range must be smaller"
                             " than maximum. Got %s." % str(self.value_range))

        X_ = to_time_series_dataset(X)
        min_t = numpy.nanmin(X_, axis=1)[:, numpy.newaxis, :]
        max_t = numpy.nanmax(X_, axis=1)[:, numpy.newaxis, :]
        range_t = max_t - min_t
        nomin = (X_ - min_t) * (self.value_range[1] - self.value_range[0])
        X_ = nomin / range_t + self.value_range[0]
        return X_


class TimeSeriesScalerMeanVariance(TransformerMixin):
    """Scaler for time series. Scales time series so that their mean (resp.
    standard deviation) in each dimension is
    mu (resp. std).

    Parameters
    ----------
    mu : float (default: 0.)
        Mean of the output time series.
    std : float (default: 1.)
        Standard deviation of the output time series.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

        NaNs within a time series are ignored when calculating mu and std.

    Examples
    --------
    >>> TimeSeriesScalerMeanVariance(mu=0.,
    ...                              std=1.).fit_transform([[0, 3, 6]])
    array([[[-1.22474487],
            [ 0.        ],
            [ 1.22474487]]])
    >>> TimeSeriesScalerMeanVariance(mu=0.,
    ...                              std=1.).fit_transform([[numpy.nan, 3, 6]])
    array([[[nan],
            [-1.],
            [ 1.]]])
    """
    def __init__(self, mu=0., std=1.):
        self.mu_ = mu
        self.std_ = std
        self.global_mean = None
        self.global_std = None

    def fit(self, X, y=None, **kwargs):
        """A dummy method such that it complies to the sklearn requirements.
        Since this method is completely stateless, it just returns itself.

        Parameters
        ----------
        X
            Ignored

        Returns
        -------
        self
        """
        return self

    def transform(self, X, **kwargs):
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
        mean_t = numpy.nanmean(X_, axis=1)[:, numpy.newaxis, :]
        std_t = numpy.nanstd(X_, axis=1)[:, numpy.newaxis, :]
        std_t[std_t == 0.] = 1.

        X_ = (X_ - mean_t) * self.std_ / std_t + self.mu_

        return X_
