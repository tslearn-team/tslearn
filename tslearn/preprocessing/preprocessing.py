from math import nan
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
    check_equal_size,
    ts_size,
    check_array,
    check_dims
)

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesResampler(TransformerMixin, TimeSeriesBaseEstimator):
    """Resampler for time series. Resample time series so that they reach the
    target size.

    Parameters
    ----------
    sz : int (default: -1)
        Size of the output time series. If not strictly positive, the size of
        the longuest timeseries in the dataset is used.

    Examples
    --------
    >>> TimeSeriesResampler(sz=5).fit_transform([[0, 3, 6]])
    array([[[0. ],
            [1.5],
            [3. ],
            [4.5],
            [6. ]]])
    """
    def __init__(self, sz: int=-1):
        self.sz = sz

    def _get_resampling_size(self, X):
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

    def _transform_unit_sz(self, X):
        n_ts, sz, d = X.shape
        X_out = numpy.empty((n_ts, 1, d))
        for i in range(X.shape[0]):
            X_out[i] = numpy.nanmean(X[i], axis=0, keepdims=True)
        return X_out

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

        target_sz = self._get_resampling_size(X_)
        if target_sz == 1:
            return self._transform_unit_sz(X_)

        n_ts, sz, d = X_.shape
        equal_size = check_equal_size(X_)
        X_out = numpy.empty((n_ts, target_sz, d))
        for i in range(X_.shape[0]):
            if not equal_size:
                sz = ts_size(X_[i])
            for di in range(d):
                X_out[i, :, di] = numpy.interp(
                    numpy.linspace(0, 1, target_sz),
                    numpy.linspace(0, 1, sz),
                    X_[i, :sz, di]
                    )
        return X_out

    def _more_tags(self):
        more_tags = super()._more_tags()
        more_tags.update({'allow_nan': True, ALLOW_VARIABLE_LENGTH: True})
        return more_tags


class TimeSeriesScalerMinMax(TransformerMixin, TimeSeriesBaseEstimator):
    """Scaler for time series datasets. Scales features values so that their span in given dimensions
    is between ``min`` and ``max`` where ``value_range=(min, max)``.

    Parameters
    ----------
    value_range : tuple (default: (0., 1.))
        The minimum and maximum value for the output time series.
    per_timeseries: bool (default: True)
        Wether the scaling should be performed per time series.
    per_feature: bool (default: True)
        Wether the scaling should be performed per feature.
        Meaningless for univariate timeseries.

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
    >>> TimeSeriesScalerMinMax(value_range=(1., 2.), per_timeseries=False, per_feature=False).fit_transform(
    ...    [[[1, 2], [2, 3]],
    ...    [[3, 4], [4, 5]]]
    ... )
    array([[[1.  , 1.25],
            [1.25, 1.5 ]],
    <BLANKLINE>
           [[1.5 , 1.75],
            [1.75, 2.  ]]])
    """
    def __init__(self, value_range=(0., 1.), per_timeseries=True, per_feature=True):
        self.value_range = value_range
        self.per_timeseries = per_timeseries
        self.per_feature = per_feature

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
        X_ = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X_)
        self._X_fit_dims = X_.shape
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled.

        Returns
        -------
        numpy.ndarray
            Resampled time series dataset.
        """
        return self.fit(X).transform(X)

    def transform(self, X, y=None, **kwargs):
        """Will normalize (min-max) each of the timeseries. IMPORTANT: this
        transformation is completely stateless, and is applied to each of
        the timeseries individually.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled.

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset.
        """
        if self.value_range[0] >= self.value_range[1]:
            raise ValueError("Minimum of desired range must be smaller"
                             " than maximum. Got %s." % str(self.value_range))

        check_is_fitted(self, '_X_fit_dims')
        X_ = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X_)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)

        axis = (1,)
        if not self.per_feature:
            axis += (2,)
        if not self.per_timeseries:
            axis += (0,)

        min_t = numpy.nanmin(X_, axis=axis, keepdims=True)
        max_t = numpy.nanmax(X_, axis=axis, keepdims=True)

        range_t = max_t - min_t
        range_t[range_t == 0.] = 1.
        nomin = (X_ - min_t) * (self.value_range[1] - self.value_range[0])
        X_ = nomin / range_t + self.value_range[0]
        return X_

    def _more_tags(self):
        more_tags = super()._more_tags()
        more_tags.update({'allow_nan': True})
        return more_tags


class TimeSeriesScalerMeanVariance(TransformerMixin, TimeSeriesBaseEstimator):
    """Scaler for time series datasets. Scales fetures values so that their mean (resp.
    standard deviation) in given dimensions is mu (resp. std).

    Parameters
    ----------
    mu : float (default: 0.)
        Mean of the output time series.
    std : float (default: 1.)
        Standard deviation of the output time series.
    per_timeseries: bool (default: True)
        Whether the scaling should be performed per time series.
    per_feature: bool (default: True)
        Whether the scaling should be performed per feature.
        Meaningless for univariate timeseries.

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
    >>> TimeSeriesScalerMeanVariance(per_timeseries=False,
    ...                              per_feature=False
    ... ).fit_transform([[[1, 2], [2, 3]], [[3, 4], [4, 5]]])
    array([[[-1.63299316, -0.81649658],
            [-0.81649658,  0.        ]],
    <BLANKLINE>
           [[ 0.        ,  0.81649658],
            [ 0.81649658,  1.63299316]]])
    """
    def __init__(self, mu=0., std=1., per_timeseries=True, per_feature=True):
        self.mu = mu
        self.std = std
        self.per_timeseries = per_timeseries
        self.per_feature = per_feature

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
        X_ = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X_)
        self._X_fit_dims = X_.shape
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be rescaled.

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
            Time series dataset to be rescaled

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        check_is_fitted(self, '_X_fit_dims')
        X_ = check_array(X, allow_nd=True, force_all_finite=False)
        X_ = to_time_series_dataset(X_)
        X_ = check_dims(X_, X_fit_dims=self._X_fit_dims, extend=False)

        axis = (1,)
        if not self.per_timeseries:
            axis += (0,)
        if not self.per_feature:
            axis += (2,)

        mean_t = numpy.nanmean(X_, axis=axis, keepdims=True)
        std_t = numpy.nanstd(X_, axis=axis, keepdims=True)

        std_t[std_t == 0.] = 1.
        X_ = (X_ - mean_t) * self.std / std_t + self.mu
        return X_

    def _more_tags(self):
        more_tags = super()._more_tags()
        more_tags.update({'allow_nan': True})
        return more_tags


class TimeSeriesImputer(TransformerMixin, TimeSeriesBaseEstimator):
    """Missing value imputer for time series.

    Missing values are replaced according to the choosen imputation method.
    There might be cases where the computation of missing values is impossible,
    in which case nans are left unchanged
    (ex: mean of all nans, ffill for the first value... ).

    Parameters
    ----------
    method : {'mean', 'median', 'ffill', 'bfill', 'constant', Callable}(default: 'mean')
        The method used to compute missing values.
        When using a Callable, the function should take an array-like
        representing a timeseries with missing values as input parameter and
        should return the transformed timeseries.
    value: float (default: nan)
        The value to replace missing values with. Only used when method is
        "constant".
    keep_trailing_nans: bool (default: True)
        Whether the trailing nans should be considered as padding for variable
        length time series and kept unprocessed. When set to false, trailing nans
        will be imputed, which can be usefull when feeding the imputer with
        :ref:`to_time_series_dataset <fun-tslearn.utils.to_time_series_dataset>`
        results.

    Notes
    -----
        This method allows datasets of variable lenght time series.
        While most missing values should be replaced, there might still be nan
        values in the resulting dataset representing padding when used with
        variable length time series.

    Examples
    --------
    >>> import math
    >>> TimeSeriesImputer().fit_transform([[0, math.nan, 6]])
    array([[[0.],
            [3.],
            [6.]]])
    >>> TimeSeriesImputer().fit_transform([[numpy.nan, 3, 6], [numpy.nan, 3]])
    array([[[4.5],
            [3. ],
            [6. ]],
    <BLANKLINE>
           [[3. ],
            [3. ],
            [nan]]])
    >>> TimeSeriesImputer('ffill').fit_transform([[[1, math.nan], [2, 3]], [[3, 4], [4, math.nan]]])
    array([[[ 1., nan],
            [ 2.,  3.]],
    <BLANKLINE>
           [[ 3.,  4.],
            [ 4.,  4.]]])
    """
    def __init__(self,
                 method: Union[str, Callable]="mean",
                 value:  Optional[float]=nan,
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

    def _more_tags(self):
        more_tags = super()._more_tags()
        more_tags.update({'allow_nan': True, ALLOW_VARIABLE_LENGTH: True})
        return more_tags
