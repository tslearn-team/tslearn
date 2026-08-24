"""Scale-predict-unscale pipeline for forecasters."""

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesMixin
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import check_array, to_time_series_dataset


class ScaledForecaster(TimeSeriesMixin, BaseEstimator):
    """Scale a series, forecast, and un-scale the forecast.

    A pattern often seen in forecasting is to scale each series before
    handing it over to a model, and to un-scale the model's predictions
    using the statistics computed by the scaler, so that the forecaster
    itself only ever sees normalized data. This cannot be expressed with a
    plain :class:`sklearn.pipeline.Pipeline`, since the last step there
    only ever calls ``predict``, never ``inverse_transform``. This estimator
    fills that gap for the ``fit(X) / predict(X=None, n=1)`` forecasters of
    :mod:`tslearn.forecasting`.

    Parameters
    ----------
        forecaster : estimator
          A tslearn forecaster, exposing ``fit(X, y=None)`` and
          ``predict(X=None, n=1)``, trained on the scaled data.
        scaler : transformer or None (default: None)
          A scaler exposing ``fit_transform``, ``transform`` and
          ``inverse_transform``, such as the ones in
          :mod:`tslearn.preprocessing`. Because ``inverse_transform`` needs
          fixed statistics to invert, ``per_timeseries`` must be set to
          False on the scaler (this is enforced at fit time). When None, a
          :class:`~tslearn.preprocessing.TimeSeriesScalerMeanVariance` with
          ``per_timeseries=False`` is used.

    Attributes
    ----------
        scaler_ : transformer
          The fitted scaler.
        forecaster_ : estimator
          The forecaster, fitted on scaled data.

    Examples
    --------
    >>> from tslearn.forecasting import VARIMA
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=2, sz=20, d=1, random_state=0)
    >>> model = ScaledForecaster(VARIMA(1, 0, 0)).fit(X)
    >>> model.predict(n=3).shape
    (2, 3, 1)

    See Also
    --------
        VARIMA, AutoVARIMA: Forecasters that can be wrapped as-is.
    """

    def __init__(self, forecaster, scaler=None):
        self.forecaster = forecaster
        self.scaler = scaler

    def _make_scaler(self):
        scaler = (
            TimeSeriesScalerMeanVariance(per_timeseries=False)
            if self.scaler is None
            else clone(self.scaler)
        )
        if getattr(scaler, "per_timeseries", False):
            raise ValueError(
                "`scaler.per_timeseries` must be False for `inverse_transform` "
                "to be available at predict time, got True."
            )
        if not hasattr(scaler, "inverse_transform"):
            raise ValueError(
                f"{scaler.__class__.__name__} does not implement "
                "`inverse_transform`, so it cannot be used to un-scale "
                "forecasts."
            )
        return scaler

    def fit(self, X, y=None):
        """Fit the scaler, then the forecaster on the scaled data.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.
            y : Ignored

        Returns
        -------
            self
                The fitted estimator
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)

        self.scaler_ = self._make_scaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.forecaster_ = clone(self.forecaster)
        self.forecaster_.fit(X_scaled)
        return self

    def predict(self, X=None, n=1):
        """Forecast ``n`` timestamps ahead, on the original data scale.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d) or None (default: None)
              Time series dataset to forecast. If None, the data passed at
              fit time is forecasted instead.
            n : int (default: 1)
              The number of timestamps to forecast, a.k.a. the horizon.

        Returns
        -------
            array of shape=(n_ts, n, d)
              Array of forecasted timestamps, on the original data scale.
        """
        check_is_fitted(self, "forecaster_")
        X_scaled = None if X is None else self.scaler_.transform(X)
        forecast_scaled = self.forecaster_.predict(X_scaled, n=n)
        return self.scaler_.inverse_transform(forecast_scaled)

    def fit_predict(self, X, y=None, n=1):
        """Fit the estimator and forecast ``n`` timestamps for the given data.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.
            y : Ignored
            n : int (default: 1)
                The number of timestamps to forecast, a.k.a. the horizon.

        Returns
        -------
            array of shape=(n_ts, n, d)
              Array of forecasted timestamps, on the original data scale.
        """
        return self.fit(X, y).predict(X, n=n)
