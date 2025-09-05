"""Min max scaler for time series preprocessing"""
import numpy

from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesBaseEstimator
from tslearn.utils import (
    to_time_series_dataset,
    check_array,
    check_dims
)

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


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
        range_t[numpy.isclose(range_t, 0.)] = 1.
        nomin = (X_ - min_t) * (self.value_range[1] - self.value_range[0])
        X_ = nomin / range_t + self.value_range[0]
        return X_

    def _more_tags(self):
        more_tags = super()._more_tags()
        more_tags.update({'allow_nan': True})
        return more_tags
