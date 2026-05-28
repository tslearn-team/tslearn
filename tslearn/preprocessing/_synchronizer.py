import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesMixin
from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.utils import check_variable_length_input, check_dims
from tslearn.utils.utils import _ts_size


class TimeSeriesFeatureSynchronizer(
    TimeSeriesMixin,
    TransformerMixin,
    BaseEstimator
):
    """
    Feature synchronizer for time series. Synchronizes features of each time series of a dataset to circumvent
    acquisition at different sampling rates or desynchronized timestamps through linear interpolation.

    Parameters
    ----------
    reference_feature_index : int (default: 0)
        The feature that is used as reference for synchronization among each time series.

    Examples
    --------
    >>> data = [
    ...    [[1, 2], [2, np.nan]],
    ...    [[1, 2], [np.nan, 3]],
    ... ]
    >>> TimeSeriesFeatureSynchronizer().fit_transform(data)
    array([[[ 1.,  2.],
            [ 2.,  2.]],
    <BLANKLINE>
           [[ 1.,  2.],
            [nan, nan]]])
    >>> data = [[[1, 2], [2, 4] , [9, np.nan]]]
    >>> timestamps = np.array([
    ...    [np.array(["2025-01-01", "2025-01-02"], dtype='datetime64'),
    ...     np.array(["2025-01-03", "2025-01-07"], dtype='datetime64'),
    ...     np.array(["2025-01-10", "nat"], dtype='datetime64')],
    ... ])
    >>> TimeSeriesFeatureSynchronizer().fit_transform(data, timestamps=timestamps)
    array([[[1. , 2. ],
            [2. , 2.4],
            [9. , 4. ]]])
    """

    def __init__(self, reference_feature_index=0):
        self.reference_feature_index = reference_feature_index

    def fit(self, X, y=None):
        """A dummy method such that it complies to the sklearn requirements.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        self
        """
        X_ = check_variable_length_input(X)
        self._X_fit_dims = X_.shape
        self.n_features_in_ = self._X_fit_dims[-1]
        return self

    def transform(self, X, y=None, timestamps=None):
        """
        Synchronizes features of each time series with the feature of reference through linear interpolation.
        When timestamps are not provided, constant sampling periods are assumed for all features and identical
        start and stop timestamps are assumed for all features of a given times series.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be synchronized feature wise.

        y:
            Ignored

        timestamps: np.datetime64 array-like of shape (n_ts, sz, d) or None (default: None)
            Acquisition timestamps, same shape as X if not None.
            When provided, timestamps should be increasing for each feature and should use
            np.datetime64('nat') for missing values.

        Returns
        -------
        numpy.ndarray
            Time series synchronized feature wise dataset.
        """
        check_is_fitted(self, '_X_fit_dims')

        X_ = check_variable_length_input(X)
        X_ = check_dims(
            X_,
            X_fit_dims=self._X_fit_dims,
            extend=False,
            check_n_features_only=True
        )

        if timestamps is not None:
            timestamps_ = check_variable_length_input(timestamps)
            if timestamps_.shape != X_.shape:
                raise ValueError("Shape mismatch between incoming data and timestamps")
            masked_timestamps = np.ma.masked_array(timestamps, mask=np.isnan(timestamps))
            if not np.ma.all(np.ma.diff(masked_timestamps, axis=1) > np.timedelta64(0)):
                # Test that valid timestamps are increasing for all TS
                raise ValueError("Timestamps must be increasing for each TS")

        ref_sizes = [_ts_size(ts[..., self.reference_feature_index]) for ts in X_]
        max_ref_size = max(ref_sizes)

        # Resize each feature
        for ts_index, ts in enumerate(X_):

            if timestamps is not None:
                evaluation_coordinates = (
                    timestamps[ts_index, :ref_sizes[ts_index], self.reference_feature_index]
                ).astype("float64")
            else:
                evaluation_coordinates = np.linspace(0, 1, ref_sizes[ts_index])

            for feature_index in range(ts.shape[-1]):
                feature_size = _ts_size(ts[..., feature_index])
                if timestamps is not None:
                    data_points_coordinates = (
                            timestamps[ts_index, :feature_size, feature_index]
                    ).astype("float64")
                else:
                    data_points_coordinates = np.linspace(0, 1, feature_size)

                ts[:ref_sizes[ts_index], feature_index] = np.interp(
                    evaluation_coordinates,
                    data_points_coordinates,
                    ts[:feature_size, feature_index]
                )
                ts[ref_sizes[ts_index]:max_ref_size, feature_index] = np.nan
        return X_[:, :max_ref_size, :]

    def fit_transform(self, X, y=None, **transform_params):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset to be synchronized feature wise.

        Returns
        -------
        numpy.ndarray
            Time series synchronized feature wise dataset.
        """
        return self.fit(X, y).transform(X, y,  **transform_params)

    def _more_tags(self):
        tags = super()._more_tags()
        tags.update({'allow_nan': True, ALLOW_VARIABLE_LENGTH: True})
        return tags

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.allow_variable_length = True
        return tags
