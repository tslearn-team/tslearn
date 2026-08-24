"""Scale-predict-unscale pipeline for forecasters."""

from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

from tslearn.backend import instantiate_backend
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

    Both scaling modes offered by :mod:`tslearn.preprocessing` scalers are
    supported. When ``scaler.per_timeseries`` is False, one scaler is fitted
    on the whole training set and its statistics are reused, unchanged, at
    predict time. When it is True (the default for tslearn scalers), each
    series is scaled with its own statistics instead.

    Parameters
    ----------
        forecaster : estimator
          A tslearn forecaster, exposing ``fit(X, y=None)`` and
          ``predict(X=None, n=1)``, trained on the scaled data.
        scaler : transformer or None (default: None)
          A scaler exposing ``fit``, ``transform`` and ``inverse_transform``,
          such as the ones in :mod:`tslearn.preprocessing`. When None, a
          :class:`~tslearn.preprocessing.TimeSeriesScalerMeanVariance` is
          used, scaling each series independently (its default,
          ``per_timeseries=True``).

    Attributes
    ----------
        per_series_ : bool
          Whether scaling is performed independently per series (mirrors
          ``scaler.per_timeseries``, defaulting to False when the scaler
          does not have such a parameter).
        scalers_ : list of transformers
          The fitted scaler(s): a single one when ``per_series_`` is False,
          one per training series otherwise.
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

    def _resolve_scaler(self):
        """Build the per-group scaler template and detect the scaling mode.

        A ``per_timeseries=True`` scaler cannot serve `inverse_transform`,
        so per-series scaling is emulated with one ``per_timeseries=False``
        clone of the scaler per series, rather than with a single
        ``per_timeseries=True`` instance for the whole batch.
        """
        scaler = (
            TimeSeriesScalerMeanVariance()
            if self.scaler is None
            else clone(self.scaler)
        )
        if not hasattr(scaler, "inverse_transform"):
            raise ValueError(
                f"{scaler.__class__.__name__} does not implement "
                "`inverse_transform`, so it cannot be used to un-scale "
                "forecasts."
            )
        per_series = bool(getattr(scaler, "per_timeseries", False))
        if per_series:
            scaler.set_params(per_timeseries=False)
        return scaler, per_series

    def _fit_scalers(self, X):
        if not self.per_series_:
            return [clone(self._scaler_template_).fit(X)]
        return [
            clone(self._scaler_template_).fit(X[i : i + 1])
            for i in range(X.shape[0])
        ]

    def _transform(self, scalers, X):
        if not self.per_series_:
            return scalers[0].transform(X)
        be = instantiate_backend(X)
        return be.vstack(
            [scaler.transform(X[i : i + 1]) for i, scaler in enumerate(scalers)]
        )

    def _inverse_transform(self, scalers, forecast):
        if not self.per_series_:
            return scalers[0].inverse_transform(forecast)
        be = instantiate_backend(forecast)
        return be.vstack(
            [
                scaler.inverse_transform(forecast[i : i + 1])
                for i, scaler in enumerate(scalers)
            ]
        )

    def fit(self, X, y=None):
        """Fit the scaler(s), then the forecaster on the scaled data.

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

        self._scaler_template_, self.per_series_ = self._resolve_scaler()
        self.scalers_ = self._fit_scalers(X)
        X_scaled = self._transform(self.scalers_, X)

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
        if X is None:
            scalers = self.scalers_
            X_scaled = None
        else:
            X = check_array(X, allow_nd=True, force_all_finite=False)
            X = to_time_series_dataset(X)
            scalers = (
                self.scalers_
                if not self.per_series_
                else self._fit_scalers(X)
            )
            X_scaled = self._transform(scalers, X)
        forecast_scaled = self.forecaster_.predict(X_scaled, n=n)
        return self._inverse_transform(scalers, forecast_scaled)

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
