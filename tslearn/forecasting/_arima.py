import warnings

from numba import njit

import numpy as np

import scipy

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import kpss

from tslearn.bases import TimeSeriesMixin, BaseModelPackage
from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.utils import check_array, to_time_series_dataset, check_dims
from tslearn.utils.utils import _to_time_series, _ts_size


def compute_var(p, X, with_constant=False):
    """Compute the VAR parameters associated with a given time series.

    Parameters
    ----------
        p : int
            AR order

        X : array-like, shape (sz, d)
            time-series data, sz should be greater or equal than p

        with_constant : bool (default: False)
            whether to compute an intercept term.

    Returns
    -------
        (intercept, ar_coeffs, residuals) tuple

    """
    X = check_array(X, ensure_min_samples=p)
    Y = X[p:]
    Z = np.array([X[t - p:t][::-1].ravel() for t in range(p, X.shape[0])])
    if with_constant:
        Z = np.hstack((
            np.ones((X.shape[0] - p, 1)),
            Z
        ))

    coeffs, *_ = np.linalg.lstsq(Z, Y, rcond=-1)

    intercept = np.array([])
    if with_constant:
        intercept = coeffs[0]

    ar_coeffs = np.array([])
    if p > 0:
        ar_coeffs = coeffs[1:] if with_constant else coeffs

    residuals = Y - np.dot(Z, coeffs)

    return intercept, ar_coeffs, residuals


@njit
def _loss(X, intercept, ar_coeffs, ma_coeffs):
    n_ts, n_samples, n_features_in = X.shape

    # Number of valid samples, discarding nans for variable length datasets
    n_valid_samples = np.sum(np.isfinite(np.sum(X, axis=-1)))

    p = ar_coeffs.shape[0]
    q = ma_coeffs.shape[0]

    sse = 0
    residuals = np.zeros((n_ts, q, n_features_in))
    start = max(p, q)
    for i in range(start, n_samples):
        current_err = X[:, i] - _varma_next(
            X[:, i - start:i],
            residuals,
            ar_coeffs,
            ma_coeffs,
            intercept,
        )
        # Deals with variable length series padded with nan
        sse += np.nansum(current_err * current_err)
        if q > 0:
            for i in range(n_ts):
                if np.all(np.isfinite(current_err[i])):
                    residuals[i, :-1] = residuals[i, 1:]
                    residuals[i, -1] = current_err[i]

    variance = sse / ((n_valid_samples - n_ts * start) * n_features_in)
    n_loglikelihood = n_features_in * n_valid_samples * (np.log(2 * np.pi) + np.log(variance) + 1) / 2
    return n_loglikelihood, residuals


@njit
def _varma_next(X, residuals, ar_coeffs, ma_coeffs, intercept):
    n_ts, n_samples, n_features_in = X.shape

    ar_forecast = np.zeros((n_ts, n_features_in))
    for k in range(ar_coeffs.shape[0]):
        ar_forecast += np.dot(X[:, n_samples - k - 1], ar_coeffs[k])
    ma_forecast = np.zeros((n_ts, n_features_in))
    for k in range(ma_coeffs.shape[0]):
        ma_forecast += np.dot(residuals[:, k], ma_coeffs[k])

    intercept = np.zeros(n_features_in) if intercept.shape[0] == 0 else intercept

    return ar_forecast + ma_forecast + np.expand_dims(intercept, axis=0)


class VARIMA(TimeSeriesMixin, BaseEstimator, BaseModelPackage):
    """
    Vector AutoRegressive Integrated Moving Average (VARIMA) estimator [1]_.

    Parameters
    ----------
        p : int, (default: 1)
          AutoRegressive (AR) order of the model.
        q : int (default: 0)
          Moving-Average (MA) order of the model.
        d : int (default: 0)
          Differentiation order of the model.
        with_constant : bool (default: True)
          Whether the model should include an intercept term.
        seasonal_period: int or None (default: None)
          When set to a positive integer :math:`m`, the model includes
          a naïve seasonal integration step where :math:`x'_t = x_t - x_{t-m}`.
        max_iter : int (default: 50)
          The maximum number of iterations used while fitting the model.

    Attributes
    ----------
        lle_ : float
          Loglikelihood of the fitted model
        intercept_ : array-like of shape=(n_features)
          Intercept term of the fitted model
        ar_coeffs_ : array-like of shape=(p, n_features, n_features)
          AR coefficients of the fitted model
        ma_coeffs_ : array-like of shape=(q, n_features, n_features)
          MA coefficients of the fitted model

    Notes
    -----
        This estimator supports variable length time-series

    See Also
    --------
        AutoVARIMA: Automatic order selection of a VARIMA model

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice. OTexts, 2014.
      https://otexts.com/fpp3/

    """

    def __init__(self, p=1, d=0, q=0, with_constant=True, seasonal_period=None, max_iter=50):
        self.p = p
        self.d = d
        self.q = q
        self.with_constant = with_constant
        self.seasonal_period = seasonal_period
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """Computes VARIMA model.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset, where the minimal value of sz depends on the p, q, d orders.
            y : Ignored

        Returns
        -------
            self
                The fitted estimator

        """
        X_ = check_array(X, allow_nd=True, force_all_finite="allow-nan")
        X_ = to_time_series_dataset(X_)

        # Minimum number of samples needed
        self._min_samples = self.p + self.q + self.d + 1 + (self.seasonal_period or 0)

        # Check min_samples
        if any(_ts_size(ts) < self._min_samples for ts in X):
            raise ValueError(f"{self._min_samples} timestamps are required per TS")

        n_ts, n_samples, n_features_in = X_.shape

        # Seasonal differencing
        s_X_ = self._seasonal_differencing(X_)
        # d-th order differencing
        d_X_ = np.diff(s_X_, n=self.d, axis=1)

        if self.q > 0:
            res = scipy.optimize.minimize(
                method="nelder-mead",
                fun=lambda params: self._loss(d_X_, params)[0],
                x0=self._default_params(d_X_),
                tol=0,
                options={"maxiter": self.max_iter},
            )
            self.intercept_, self.ar_coeffs_, self.ma_coeffs_ = self._unravel_params(res.x, n_features_in)
        else:
            self.intercept_, self.ar_coeffs_, self.ma_coeffs_ = self._unravel_params(
                self._default_params(d_X_),
                n_features_in
            )

        # Last residuals and last timestamps needed to predict fitted data
        n_lle, self._last_residuals = self._loss(d_X_)
        self._last_timestamps = self._get_last_timestamps(X_)

        self.lle_ = -n_lle
        self.n_ts_ = n_ts
        self.n_samples_ = n_samples
        self.n_features_in_ = n_features_in

        return self

    def _seasonal_differencing(self, X):
        if self.seasonal_period is not None:
            seasonal_differenced_X = (
                X[:, self.seasonal_period:] -
                X[:, :-self.seasonal_period]
            )
        else:
            seasonal_differenced_X = X
        return seasonal_differenced_X

    def _get_last_timestamps(self, X):
        n_ts, sz, n_features_in = X.shape
        if max(self.p, self.q, self.d, self.seasonal_period or 0):
            # Move nans of variable length time series up front to properly select last_values
            for i, ts in enumerate(X):
                X[i] = np.roll(ts, sz - _ts_size(ts), axis=0)
            n_timestamp_to_keep = (self.seasonal_period or 0) + self.d + self.p
            last_timestamps = X[:, -n_timestamp_to_keep:]
        else:
            # No need to keep any
            last_timestamps = np.empty((n_ts, 0, n_features_in))
        return last_timestamps

    def _default_params(self, X):
        params = np.vstack([
            self._default_params_per_ts(_to_time_series(ts, remove_nans=True)) for ts in X
        ])
        return np.mean(params, axis=0)

    def _default_params_per_ts(self, ts):
        # Initialize ar_coeffs with VAR results
        constant_params, ar_start_params, residuals = compute_var(self.p, ts, self.with_constant)
        # Initialize ma_coeffs with VAR results on residuals of previous step
        ma_start_params = compute_var(self.q, residuals)[1]

        return np.concatenate((constant_params, ar_start_params.T.ravel(), ma_start_params.T.ravel()))

    def _unravel_params(self, params, n_features_in=None):
        n_features_in = n_features_in if n_features_in is not None else self.n_features_in_

        if self.with_constant:
            intercept = params[:n_features_in]
            params = params[n_features_in:]
        else:
            intercept = np.array([])
        if self.p > 0:
            ar_coeffs = params[:n_features_in * (n_features_in * self.p)]
            ar_coeffs = ar_coeffs.reshape(self.p, n_features_in, n_features_in)
            params = params[n_features_in * (n_features_in * self.p):]
        else:
            ar_coeffs = np.empty((0, n_features_in, n_features_in))
        if self.q > 0:
            ma_coeffs = params[:n_features_in * (n_features_in * self.q)]
            ma_coeffs = ma_coeffs.reshape(self.q, n_features_in, n_features_in)
        else:
            ma_coeffs = np.empty((0, n_features_in, n_features_in))

        return intercept, ar_coeffs, ma_coeffs

    def _loss(self, X, params=None):
        n_features_in = X.shape[-1]

        if params is not None:
            intercept, ar_coeffs, ma_coeffs = self._unravel_params(params, n_features_in)
        else:
            intercept, ar_coeffs, ma_coeffs = self.intercept_, self.ar_coeffs_, self.ma_coeffs_
        return _loss(X, intercept, ar_coeffs, ma_coeffs)

    def _undifference(self, initial_values, prediction):
        _ = np.array([(-1 ** (j+1)) * scipy.special.binom(j + 1, self.d) for j in range(self.d)])
        res = prediction - np.dot(_, initial_values[:, -self.d:])
        return res

    def predict(self, X=None, n=1):
        """Forecasts n timestamps of the given data if any, otherwise forecasts n timestamps
        for the fitted data.

        Parameters
        ----------
            X : array-like, shape (n_ts, sz, d), optional
              Time-series dataset to forecast. If None, the fitted data is forecasted
              otherwise the fitted model is applied to the given data.
            n : int (default: 1)
              The number of timestamps to forecast, a.k.a. the horizon.

        Returns
        -------
            array, shape = (n_ts, n, d)
              Array of forecasted timestamps

        """
        check_is_fitted(self)

        if X is None:
            last_timestamps = self._last_timestamps
            last_residuals = self._last_residuals

        else:
            X_ = check_array(X, allow_nd=True, force_all_finite="allow-nan")
            X_ = to_time_series_dataset(X_)

            # Check min_samples
            if any(_ts_size(ts) < self._min_samples for ts in X):
                raise ValueError(f"{self._min_samples} timestamps are required per TS")

            # Check number of features
            check_dims(X_, (0, 0, self.n_features_in_), check_n_features_only=True)

            # Retrieve last residuals and last timestamps needed to compute estimate
            last_timestamps = self._get_last_timestamps(X_)
            seasonal_differrenced_X = self._seasonal_differencing(X_)
            differenced_X = np.diff(seasonal_differrenced_X, n=self.d, axis=1)
            last_residuals = self._loss(differenced_X)[1]

        # Seasonal differencing
        sd_timestamps = self._seasonal_differencing(last_timestamps)
        # d-th order differencing
        dd_timestamps = np.diff(sd_timestamps, n=self.d, axis=1)

        res = np.zeros((last_timestamps.shape[0], n, self.n_features_in_))
        for i in range(n):
            estimate = _varma_next(
                dd_timestamps,
                last_residuals,
                self.ar_coeffs_,
                self.ma_coeffs_,
                self.intercept_
            )
            if self.p > 0 and n > 1:
                # Rolling for next estimate
                dd_timestamps = np.roll(dd_timestamps, -1, axis=1)
                dd_timestamps[:, -1] = estimate
            if self.q > 0 and n > 1:
                # Rolling for next estimate
                last_residuals = np.roll(last_residuals, -1, axis=1)
                last_residuals[:, -1] = np.zeros(self.n_features_in_)
            if self.d:
                estimate = self._undifference(sd_timestamps, estimate)
                if n > 1:
                    # Rolling for next estimate
                    sd_timestamps = np.roll(sd_timestamps, -1, axis=1)
                    sd_timestamps[:, -1] = estimate
            if self.seasonal_period:
                estimate = estimate + last_timestamps[:, -self.seasonal_period]
                if n > 1:
                    # Rolling for next estimate
                    last_timestamps = np.roll(last_timestamps, -1, axis=1)
                    last_timestamps[:, -1] = estimate
            res[:, i] = estimate

        return res

    def fit_predict(self, X, y=None, n=1):
        """Computes VARIMA model and forecasts n timestamps for the given data.

        Parameters
        ----------
            X: array-like, shape (n_ts, sz, d)
              Time-series dataset.
            y : Ignored
            n : int (default: 1)
              The number of timestamps to forecast, a.k.a. the horizon.

        Returns
        -------
            array, shape = (n_ts, n, d)
              Array of forecasted timestamps

        """
        self.fit(X, None)
        return self.predict(n=n)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.allow_variable_length = True
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        tags = super()._more_tags()
        tags.update({
            "requires_y": False,
            "allow_nan": True,
            ALLOW_VARIABLE_LENGTH: True
        })
        return tags


class AutoVARIMA(TimeSeriesMixin, BaseEstimator, BaseModelPackage):
    """Automatically selects the best Vector Autoregressive Moving Average (VARIMA) model
    through Hyndman-Khandakar algorithm [1]_ [2]_.

    Parameters
    ----------
    max_p : int (default: 5)
        Maximum order of the AutoRegressive (AR) component to consider.
    max_d : int (default: 2)
        Maximum order of differencing considered to achieve stationarity.
    max_q : int (default: 5)
        Maximum order of the Moving-Average (MA) component to consider.
    default_d_for_non_stationarity : int or None (default 0)
        Used as differentiation order if stationarity cannot be achieved
        within max_d. If None, an error is raised if stationarity cannot
        be achieved within max_d.
    seasonal_period: int or None (default: None)
        Naïve seasonal integration to apply to VARIMA models

    Attributes
    ----------
        best_estimator_: VARIMA
            the fitted VARIMA model.

    Notes
    -----
    This estimator supports variable length time-series

    See Also
    --------
        VARIMA: Vector AutoRegressive Integrated Moving Average (VARIMA) estimator.

    References
    ----------
    .. [1] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice. OTexts, 2014.
      https://otexts.com/fpp3/
    .. [2] Hyndman, R. J., & Khandakar, Y. (2008).
      Automatic time series forecasting: The forecast package for R. Journal of Statistical Software, 27(1), 1–22.

    """

    def __init__(self, max_p=5, max_d=2, max_q=5, default_d_for_non_stationarity=0, seasonal_period=None):
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.default_d_for_non_stationarity = default_d_for_non_stationarity
        self.seasonal_period = seasonal_period

    def fit(self, X, y=None):
        """Selects the best Vector Autoregressive Moving Average (VARIMA) model
        through Hyndman-Khandakar algorithm for the given data.

        Parameters
        ----------
            X: array-like, shape (n_ts, sz, d)
              Time-series dataset.
            y : Ignored

        Returns
        -------
            self
                the fitted estimator

        """
        X_ = check_array(X, allow_nd=True, force_all_finite="allow-nan")
        X_ = to_time_series_dataset(X_)

        # Minimum samples needed for seasonal integration for kpss test
        self._min_samples = self.seasonal_period or 0

        # Check min_samples
        if any(_ts_size(ts) < self._min_samples for ts in X):
            raise ValueError(f"{self._min_samples} timestamps are required per TS")

        self.best_estimator_ = self._get_best_model_for_given_d(self._determine_d(X_), X_)
        self.best_estimator_.fit(X_)

        self.n_ts_ = self.best_estimator_.n_ts_
        self.n_samples_ = self.best_estimator_.n_samples_
        self.n_features_in_ = self.best_estimator_.n_features_in_

        return self

    def fit_predict(self, X, y=None,  n=1):
        """Selects the best Vector Autoregressive Moving Average (VARIMA) model
        and forecasts n timestamps for the given data.

        Parameters
        ----------
            X: array-like, shape (n_ts, sz, d)
              Time-series dataset.
            y : Ignored
            n : int (default: 1)
              The number of timestamps to forecast, a.k.a. the horizon.

        Returns
        -------
            array, shape = (n_ts, n, d)
              Array of forecasted timestamps

        """
        self.fit(X, None)
        return self.predict(n=n)

    def predict(self, X=None, n=1):
        """Forecast with the model selected at fitting time.

        Parameters
        ----------
            X : array-like, shape (n_ts, sz, d), optional
              Time-series dataset to forecast. If None, the fitted data is forecasted
              otherwise the fitted model is applied to the given data.
            n : int (default: 1)
              The number of timestamps to forecast, a.k.a. the horizon.

        Returns
        -------
            array, shape = (n_ts, n, d)
              Array of forecasted timestamps

        """
        check_is_fitted(self)
        return self.best_estimator_.predict(X, n)

    @staticmethod
    def _is_stationary(X):
        # Check stationarity for each feature of each TS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=InterpolationWarning)
            return all(
                kpss(_to_time_series(x, remove_nans=True), regression='c')[1] > 0.05
                for k in range(X.shape[-1])
                for x in X[:, :, k]
            )

    def _determine_d(self, X):
        # Apply naïve seasonality first
        if self.seasonal_period is not None:
            seasonal_differenced_X = (
                    X[:, self.seasonal_period:] -
                    X[:, :-self.seasonal_period]
            )
        else:
            seasonal_differenced_X = X

        d = 0
        while not self._is_stationary(np.diff(seasonal_differenced_X, d, axis=1)):
            d += 1
            if d > self.max_d:
                if self.default_d_for_non_stationarity is not None:
                    return self.default_d_for_non_stationarity
                raise ValueError("Maximum differencing order reached")
        return d

    def _init_models(self, d):
        models = {
            VARIMA(0, d, 0, with_constant=d < 2, seasonal_period=self.seasonal_period)
        }
        if self.max_p > 0:
            models.add(VARIMA(1, d, 0, with_constant=d < 2, seasonal_period=self.seasonal_period))
        if self.max_q > 0:
            models.add(VARIMA(0, d, 1, with_constant=d < 2, seasonal_period=self.seasonal_period))
        if self.max_p > 1 and self.max_q > 1:
            models.add(VARIMA(2, d, 2, with_constant=d < 2, seasonal_period=self.seasonal_period),)
        if d <= 1:
            models.add(VARIMA(0, d, 0, with_constant=False, seasonal_period=self.seasonal_period))
        return models

    def _compute_model_variations(self, current_best_model, computed_adjustable_hyperparams):
        p_variations = [current_best_model.p]
        if not current_best_model.p + 1 > self.max_p:
            p_variations.append(current_best_model.p + 1)
        if current_best_model.p - 1 >= 0:
            p_variations.append(current_best_model.p - 1)

        q_variations = [current_best_model.q]
        if not current_best_model.q + 1 > self.max_q:
            q_variations.append(current_best_model.q + 1)
        if current_best_model.q - 1 >= 0:
            q_variations.append(current_best_model.q - 1)

        eligible_variations = {(current_best_model.p, current_best_model.q, not current_best_model.with_constant)}
        for p in p_variations:
            for q in q_variations:
                eligible_variations.add((p, q,  current_best_model.with_constant))

        eligible_variations = eligible_variations - computed_adjustable_hyperparams

        return {
            VARIMA(p, current_best_model.d, q, with_constant, self.seasonal_period)
            for p, q, with_constant in eligible_variations
        }

    def _get_best_model_for_given_d(self, d, X):
        adjustable_hyperparams = ["p", "q", "with_constant"]
        best_model = None
        minimal_aic = np.inf
        computed_adjustable_hyperparams = set()

        models_to_test = self._init_models(d)

        while models_to_test:
            for model in models_to_test:
                model_adjustable_hyperparams = tuple(getattr(model, hyperparam) for hyperparam in adjustable_hyperparams)
                computed_adjustable_hyperparams.add(model_adjustable_hyperparams)

                try:
                    model = model.fit(X)
                except ValueError as e:
                    warnings.warn(f"Model {model} skipped: {e}")
                    continue

                model_aic = 2 * (model.p + model.q + (2 if model.with_constant else 1) - model.lle_)
                if model_aic < minimal_aic:
                    best_model = model
                    minimal_aic = model_aic

            models_to_test = self._compute_model_variations(best_model, computed_adjustable_hyperparams)

        return best_model

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.allow_variable_length = True
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        tags = super()._more_tags()
        tags.update({
            "requires_y": False,
            "allow_nan": True,
            ALLOW_VARIABLE_LENGTH: True
        })
        return tags
