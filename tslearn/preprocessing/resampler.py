"""Resampler for time series preprocessing"""
import math
import typing

import numpy

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesBaseEstimator
from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.utils import (
    check_variable_length_input,
    check_equal_size,
    ts_size,
    check_dims
)

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesResampler(TransformerMixin, TimeSeriesBaseEstimator):
    """Resampler for time series. Resample time series so that they reach the
    target size.

    Each time series of a dataset is processed independently. If the target size
    is equal to the actual size of the sime series, the time series is left
    unchanged. When computing the actual size of a time series, trailing empty
    samples (nans on all dimensions) are not taken into account.

    Parameters
    ----------
    sz : int (default: -1)
        Size of the output time series. If not strictly positive, the size of
        the longuest time series in the dataset is used.
    method: {'linear', 'mean', 'max', 'uniform'} (default: linear)
        The method used to resample.
        - linear: linear interpolation for `sz` evenly spaced samples among
        each time series.
        - mean: mean computed for `sz` evenly spaced samples among each time
         series within a `window_size` interval.
        - max: mean computed for `sz` evenly spaced samples among each time
         series within a `window_size` interval.
        - uniform: select `sz` evenly spaced samples among each time series

    window_size: strictly positive int or None (default: None)
        Custom window size. Used for `mean` and `max` resampling method. Ignored
        otherwise. If set to `None`, the resampling factor is used.

    Examples
    --------
    >>> # Linear upsampling
    >>> TimeSeriesResampler(sz=5).fit_transform([[0, 3, 6]])
    array([[[0. ],
            [1.5],
            [3. ],
            [4.5],
            [6. ]]])
    >>> # Linear downsampling
    >>> TimeSeriesResampler(sz=5).fit_transform([[0, 3, 6, 9, 12, 15]])
    array([[[ 0.  ],
            [ 3.75],
            [ 7.5 ],
            [11.25],
            [15.  ]]])
    >>> # Mean downsampling with custom window size
    >>> TimeSeriesResampler(sz=5, method="mean", window_size=2).fit_transform([[0, 3, 6, 9, 12, 15]])
    array([[[ 1.5],
            [ 4.5],
            [ 7.5],
            [10.5],
            [13.5]]])
    """
    def __init__(self,
                 sz: int = -1,
                 method: str = 'linear',
                 window_size: typing.Optional[int] = None) -> None:
        self.sz = sz
        self.method = method
        self.window_size = window_size

    @property
    def _resampler(self) -> typing.Optional[typing.Callable]:
        return getattr(self, "_{}_resample".format(self.method), None)

    def _get_resampling_size(self, X) -> int:
        return self.sz if self.sz > 0 else X.shape[1]

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
        X_ = check_variable_length_input(X)
        self._X_fit_dims = X_.shape

        return self

    def _linear_resample(self, X):
        target_sz = self._get_resampling_size(X)
        if target_sz == 1:
            return numpy.nanmean(X, axis=1, keepdims=True)

        n_ts, sz, d = X.shape
        equal_size = check_equal_size(X)
        X_out = numpy.empty((n_ts, target_sz, d))
        for i in range(X.shape[0]):
            if not equal_size:
                sz = ts_size(X[i])
            for di in range(d):
                X_out[i, :, di] = numpy.interp(
                    numpy.linspace(0, 1, target_sz),
                    numpy.linspace(0, 1, sz),
                    X[i, :sz, di]
                    )
        return X_out

    def _uniform_resample(self, X):
        target_size = self._get_resampling_size(X)
        n_ts, sz, d = X.shape
        equal_size = check_equal_size(X)
        X_out = numpy.empty((n_ts, target_size, d))
        for i in range(X.shape[0]):
            if not equal_size:
                sz = ts_size(X[i])
            indices = numpy.rint(numpy.linspace(0, sz -1, target_size)).astype("int64")
            X_out[i] = X[i, indices]
        return X_out

    def _max_resample(self, X):
        return self._window_resample_generic(X, numpy.max)

    def _mean_resample(self, X):
        return self._window_resample_generic(X, numpy.nanmean)

    def _window_resample_generic(self, X, method):
        target_size = self._get_resampling_size(X)
        if target_size == 1:
            return method(X, axis=1, keepdims=True)

        n_ts, sz, d = X.shape
        equal_size = check_equal_size(X)
        X_out = numpy.zeros((n_ts, target_size, d))
        for i in range(X.shape[0]):
            if not equal_size:
                sz = ts_size(X[i])
            if sz == target_size:
                X_out[i] = X[i, :sz]
            else:
                self._window_resample(X[i], X_out[i], method)
        return X_out

    def _window_resample(self, timeseries, output, method) -> None:
        original_size = ts_size(timeseries)
        target_size = self._get_resampling_size(timeseries)
        window_size = self.window_size or max(target_size / original_size, original_size / target_size)
        indices = numpy.linspace(0, original_size - 1, target_size)
        for output_index, input_index in enumerate(indices):
            output[output_index] = self._compute_window(timeseries, input_index, method, window_size)

    @staticmethod
    def _compute_window(timeseries, index, method, window_size):
        return method(
            timeseries[max(0, math.ceil(index - window_size/2)): math.floor(index + window_size/2) + 1],
            axis=0,
            keepdims=True)

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be resampled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        return self.fit(X).transform(X)

    def transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be resampled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        check_is_fitted(self, '_X_fit_dims')

        X_ = check_variable_length_input(X)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)

        resampler = self._resampler
        if resampler is None:
            raise ValueError("Resampler {} not implemented.".format(self.method))

        return resampler(X_)

    def _more_tags(self) -> dict[str, typing.Any]:
        more_tags = super()._more_tags()
        more_tags.update({'allow_nan': True, ALLOW_VARIABLE_LENGTH: True})
        return more_tags
