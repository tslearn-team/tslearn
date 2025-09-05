"""Imputer for time series preprocessing"""
from typing import Callable, Optional, Union

import numpy

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesBaseEstimator
from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.utils import (
    check_variable_length_input,
    to_time_series_dataset,
    to_time_series,
    ts_size,
    check_dims
)

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesImputer(TransformerMixin, TimeSeriesBaseEstimator):
    """Missing value imputer for time series.

    Missing values (nans) are replaced according to the chosen imputation
    method. There might be cases where the computation of missing values is
    impossible, in which case they are left unchanged
    (ex: mean of all nans, ffill for the first value... ).

    The imputer can be configured so that trailing 'empty' samples (nans for
    all features) are unprocessed by setting the `keep_trailing_nans` parameter
    to `True`. This might be handy when dealing with variable length time
    series datasets formatted with
    :ref:`to_time_series_dataset <fun-tslearn.utils.to_time_series_dataset>`,
    where time series are padded with 'empty' samples to match the length of the
    longest time serie. This option aims at preserving the variable length
    nature of the input dataset.

    Time series are processed sequentially by the :func:`~transform` and
    :func:`~fit_transform` methods, and gathered using
    :ref:`to_time_series_dataset <fun-tslearn.utils.to_time_series_dataset>`,
    effectively padding if needed.

    Parameters
    ----------
    method : {'mean', 'median', 'ffill', 'bfill', 'linear', 'constant', Callable}(default: 'mean')
        The method used to compute missing values.

        When using linear imputation, starting nans will be replaced with first non-null value
        and ending nans will be replaced with last non-null value (
        except for 'empty' samples when `keep_trailing_nans` set to `True`).

        When using a Callable, the function should take an array-like
        representing a timeseries with missing values as input parameter and
        should return the transformed timeseries.
    value: float (default: nan)
        The value to replace missing values with. Only used when method is
        `constant`.
    keep_trailing_nans: bool (default: False)
        Whether trailing samples with nans on all dimensions should be considered
        padding for variable length time series and kept unprocessed. When set to
        `True`, trailing 'empty' samples  will not be imputed.

    Notes
    -----
        This method allows datasets of variable lenght time series.
        While most missing values should be replaced, there might still be nan
        values in the resulting dataset representing padding when used with
        variable length time series, or uncomputable data.

    Examples
    --------
    >>> TimeSeriesImputer().fit_transform([[0, numpy.nan, 6]])
    array([[[0.],
            [3.],
            [6.]]])
    >>> # Padding occurs after processing for variable length inputs
    >>> TimeSeriesImputer().fit_transform([[numpy.nan, 3, 6], [numpy.nan, 3]])
    array([[[4.5],
            [3. ],
            [6. ]],
    <BLANKLINE>
           [[3. ],
            [3. ],
            [nan]]])
    >>> # Trailing empty samples are preserved with `keep_trailing_nans`
    >>> TimeSeriesImputer('ffill', keep_trailing_nans=True).fit_transform(
    ... [[[1, 2], [2, numpy.nan]], [[3, 4], [numpy.nan, numpy.nan]]]
    ... )
    array([[[ 1.,  2.],
            [ 2.,  2.]],
    <BLANKLINE>
           [[ 3.,  4.],
            [nan, nan]]])
    >>> # Uncomputable values are left unchanged
    >>> TimeSeriesImputer('ffill').fit_transform([[numpy.nan, 3, 6]])
    array([[[nan],
            [ 3.],
            [ 6.]]])
    """
    def __init__(self,
                 method: Union[str, Callable]="mean",
                 value:  Optional[float]=numpy.nan,
                 keep_trailing_nans: bool = False):
        self.method = method
        self.value = value
        self.keep_trailing_nans = keep_trailing_nans
        super().__init__()

    @property
    def _imputer(self):
        if callable(self.method):
            return self.method

        if hasattr(self, "_{}_impute".format(self.method)):
            return getattr(self, "_{}_impute".format(self.method))
        return None

    def _constant_impute(self, ts):
        return numpy.where(numpy.isnan(ts), self.value, ts)

    @staticmethod
    def _mean_impute(ts):
        return numpy.where(numpy.isnan(ts), numpy.nanmean(ts, axis=0, keepdims=True), ts)

    @staticmethod
    def _median_impute(ts):
        return numpy.where(numpy.isnan(ts), numpy.nanmedian(ts, axis=0, keepdims=True), ts)

    @staticmethod
    def _linear_impute(ts):
        for di in range(ts.shape[-1]):
            ts_di = ts[:, di]
            mask = numpy.isnan(ts_di)
            ts_di[mask] = numpy.interp(
                numpy.nonzero(mask)[0],
                numpy.nonzero(~mask)[0],
                ts_di[~mask]
            )
        return ts

    @staticmethod
    def _ffill_impute(ts):
        # Forward fill
        mask = numpy.isnan(ts)
        idx = numpy.where(
            ~mask,
            numpy.arange(ts.shape[0]).reshape(ts.shape[0], 1),
            0
        )
        numpy.maximum.accumulate(idx, axis=0, out=idx)
        if ts.shape[-1] > 1:
            # Multivariate
            ts[mask] = ts[idx[mask], numpy.nonzero(mask)[1]]
        else:
            # Univariate
            ts[mask] = ts[idx[mask]].flatten()
        return ts

    @staticmethod
    def _bfill_impute(ts):
        # Backward fill
        mask = numpy.isnan(ts)
        idx = numpy.where(
            ~mask,
            numpy.arange(ts.shape[0]).reshape(ts.shape[0], 1),
            ts.shape[0] -1
        )
        numpy.minimum.accumulate(numpy.flip(idx, axis=0), axis=0, out=idx)
        idx = numpy.flip(idx, axis=0)
        if ts.shape[-1] > 1:
            # Multivariate
            ts[mask] = ts[idx[mask], numpy.nonzero(mask)[1]]
        else:
            # Univariate
            ts[mask] = ts[idx[mask]].flatten()
        return ts

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

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be imputed.

        Returns
        -------
        numpy.ndarray
            Imputed time series dataset.
        """
        return self.fit(X).transform(X, kwargs)

    def transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be imputed

        Returns
        -------
        numpy.ndarray
            Imputed time series dataset
        """
        check_is_fitted(self, '_X_fit_dims')

        X_ = check_variable_length_input(X)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)

        imputer = self._imputer
        if imputer is None:
            raise ValueError("Imputer {} not implemented.".format(self.method))

        for ts_index in range(X_.shape[0]):
            ts = to_time_series(X[ts_index])
            stop_index = ts.shape[0]
            if self.keep_trailing_nans:
                stop_index = ts_size(ts)
            X_[ts_index, :stop_index] = imputer(ts[:stop_index])
        return to_time_series_dataset(X_)

    def _more_tags(self) -> dict:
        more_tags = super()._more_tags()
        more_tags.update({'allow_nan': True, ALLOW_VARIABLE_LENGTH: True})
        return more_tags
