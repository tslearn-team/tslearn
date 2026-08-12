"""Re-use of pre-trained time series models for forecasting."""

import inspect
import warnings

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesMixin
from tslearn.utils import check_array, to_time_series_dataset

from ._embedding import (
    LAYOUTS,
    TimeSeriesFoundationEmbedder,
    _check_probe_pooling,
    _layout_to_model_input,
    _require_torch,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


#: Method names commonly exposed by pre-trained forecasters, by decreasing
#: order of preference.
CANDIDATE_PREDICT_METHODS = (
    "predict",
    "forecast",
    "predict_quantiles",
    "generate",
)

#: Names of the argument through which a forecast horizon is commonly passed.
CANDIDATE_HORIZON_NAMES = (
    "prediction_length",
    "horizon",
    "forecast_horizon",
    "prediction_horizon",
    "num_steps",
    "n_steps",
    "steps",
    "h",
    "n",
)

#: Attribute names under which forecasts are commonly returned.
CANDIDATE_FORECAST_NAMES = (
    "quantile_preds",
    "prediction_outputs",
    "predictions",
    "sequences",
    "mean",
)


#: Ways of presenting a batch of univariate contexts to a model, by decreasing
#: order of preference. Implementations disagree on this: Chronos-Bolt expects a
#: ``(batch, length)`` array while Chronos-2 requires an explicit variate axis,
#: and GluonTS-derived pipelines take a list of series. The first representation
#: the model accepts is used, and remembered in the ``context_format_``
#: attribute.
CONTEXT_FORMATS = ("2d", "3d", "list")


def _format_context(context, context_format):
    """Present a ``(n_rows, sz)`` batch of univariate contexts to a model."""
    if context_format == "2d":
        return context
    if context_format == "3d":
        # (n_rows, n_variates=1, sz)
        return context[:, None, :]
    if context_format == "list":
        return [row for row in context]
    raise ValueError(  # pragma: no cover
        f"`context_format` must be one of {CONTEXT_FORMATS}, got "
        f"'{context_format}'."
    )


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32).cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _unwrap_forecast(output):
    """Extract a numeric array out of an arbitrary forecasting output."""
    for name in CANDIDATE_FORECAST_NAMES:
        value = getattr(output, name, None)
        if value is not None and not callable(value):
            return _to_numpy(value)
    if isinstance(output, dict):
        for name in CANDIDATE_FORECAST_NAMES:
            if name in output:
                return _to_numpy(output[name])
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("The model returned an empty forecast.")
        items = [_to_numpy(item) for item in output]
        shapes = {item.shape for item in items}
        if len(shapes) == 1:
            # A list with one entry per series, as returned e.g. by pipelines
            return np.stack(items, axis=0)
        # A tuple of heterogeneous outputs: keep the first one
        return items[0]
    return _to_numpy(output)


def _resolve_horizon_axis(shape, horizon, horizon_axis):
    """Locate the axis of a forecast array that holds the forecast horizon."""
    ndim = len(shape)
    if horizon_axis != "auto":
        axis = horizon_axis + ndim if horizon_axis < 0 else horizon_axis
        if not 1 <= axis < ndim:
            raise ValueError(
                f"`horizon_axis={horizon_axis}` is out of range for a forecast "
                f"of shape {shape}; it must designate one of the axes 1 to "
                f"{ndim - 1}, the axis 0 being the series."
            )
        if shape[axis] < horizon:
            raise ValueError(
                f"`horizon_axis={horizon_axis}` designates an axis of size "
                f"{shape[axis]} in a forecast of shape {shape}, which cannot "
                f"hold a horizon of {horizon}."
            )
        return axis

    # An axis whose size is exactly the horizon is a much safer guess than one
    # that is merely large enough; among those, the last one is retained, as
    # forecasts are most often returned with time as their trailing axis.
    exact_axes = [axis for axis in range(1, ndim) if shape[axis] == horizon]
    if exact_axes:
        ambiguous = [axis for axis in exact_axes if shape[axis] > 1]
        if len(ambiguous) > 1:
            warnings.warn(
                f"The forecast returned by the model has shape {shape}, in "
                f"which several axes could be the horizon axis of length "
                f"{horizon}; axis {exact_axes[-1]} was assumed. Set "
                "`horizon_axis` explicitly to remove the ambiguity.",
                UserWarning,
                stacklevel=3,
            )
        return exact_axes[-1]

    longer_axes = [axis for axis in range(1, ndim) if shape[axis] > horizon]
    if not longer_axes:
        raise ValueError(
            f"Could not identify the forecast horizon axis in an output of "
            f"shape {shape} for a horizon of {horizon}. Set `horizon_axis` "
            "explicitly, or pass a `predict_fn` to control how the model is "
            "called."
        )
    return min(longer_axes, key=lambda axis: shape[axis])


def _reduce_to_point_forecast(forecast, n_rows, horizon, quantile, horizon_axis):
    """Reduce an arbitrarily shaped univariate forecast to (n_rows, horizon).

    Any axis that is neither the batch axis nor the horizon axis is understood
    as holding quantile levels or sample paths, and is therefore reduced.
    """
    forecast = np.asarray(forecast, dtype=np.float64)
    if forecast.shape[0] != n_rows:
        raise ValueError(
            f"The model returned forecasts for {forecast.shape[0]} series while "
            f"{n_rows} were provided."
        )
    if forecast.ndim == 1:  # pragma: no cover
        forecast = forecast[:, None]
    if forecast.ndim == 2:
        if forecast.shape[1] < horizon:
            raise ValueError(
                f"The model returned forecasts of length {forecast.shape[1]}, "
                f"which is shorter than the requested horizon {horizon}."
            )
        return forecast[:, :horizon]

    axis = _resolve_horizon_axis(forecast.shape, horizon, horizon_axis)
    forecast = np.moveaxis(forecast, axis, -1)
    forecast = forecast.reshape(n_rows, -1, forecast.shape[-1])
    if forecast.shape[1] == 1:
        forecast = forecast[:, 0]
    elif quantile is None:
        forecast = forecast.mean(axis=1)
    else:
        forecast = np.quantile(forecast, quantile, axis=1)
    return forecast[:, :horizon]


class _BaseFoundationForecaster(TimeSeriesMixin, BaseEstimator):
    """Shared input handling for pre-trained forecasters."""

    def _check_input(self, X):
        X = check_array(X, allow_nd=True, force_all_finite=True)
        return to_time_series_dataset(X)

    def _check_layout(self):
        if self.input_layout not in LAYOUTS:
            raise ValueError(
                f"`input_layout` must be one of {LAYOUTS}, got "
                f"'{self.input_layout}'."
            )
        horizon_axis = getattr(self, "horizon_axis", "auto")
        if horizon_axis != "auto" and not isinstance(
            horizon_axis, (int, np.integer)
        ):
            raise ValueError(
                f"`horizon_axis` must be an integer or 'auto', got "
                f"{horizon_axis!r}."
            )

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
              Array of forecasted timestamps

        """
        return self.fit(X, y).predict(X, n=n)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.input_tags.allow_nan = False
        return tags


class ZeroShotForecaster(_BaseFoundationForecaster):
    """Forecast with a pre-trained model, without any training.

    Time series foundation models are pre-trained on large corpora of series
    and are meant to forecast unseen series out of the box. This estimator
    wraps such a model behind the usual tslearn forecasting API, so that it can
    be compared to, or combined with, the other estimators of the library.
    Calling :meth:`fit` is therefore not required and does not learn anything.

    The estimator makes no assumption about a particular implementation. It
    only assumes that ``model`` exposes a forecasting method (auto-detected
    among ``predict_quantiles``, ``predict``, ``forecast`` and ``generate``)
    accepting the context values as its first argument and a forecast horizon
    as a keyword argument. When this is not the case, or when finer control is
    needed, pass an explicit ``predict_fn``.

    Parameters
    ----------
        model : object
          A pre-trained forecasting model or inference pipeline.
        predict_fn : callable or None (default: None)
          Called as ``predict_fn(model, context, horizon)``, where ``context``
          is a NumPy array laid out as prescribed by ``input_layout``, and
          expected to return forecasts that can be broadcast to
          ``(n_rows, horizon)``. When None, the model's own forecasting method
          is auto-detected.
        input_layout : {"univariate", "channels_last", "channels_first"} (default: "univariate")
          How the context is laid out when handed over to the model.
          ``"univariate"`` forecasts every channel of every series
          independently and feeds a ``(n_ts * d, sz)`` array, which is what
          most time series foundation models expect. The other two layouts feed
          a ``(n_ts, sz, d)`` or ``(n_ts, d, sz)`` array respectively, for
          models that are natively multivariate.
        context_length : int or None (default: None)
          When set, only the last ``context_length`` timestamps of each series
          are used as context.
        horizon_axis : int or "auto" (default: "auto")
          Which axis of the array returned by the model holds the forecast
          horizon, axis 0 being the series. There is no shared convention here:
          Chronos-2 returns ``(n_series, n_quantiles, horizon)`` while its own
          ``predict_quantiles`` returns ``(n_series, horizon, n_quantiles)``,
          so setting this explicitly, e.g. ``horizon_axis=-1``, is the reliable
          option. When ``"auto"``, the axis is inferred from the returned
          shape, preferring an axis whose size is exactly the horizon and, among
          those, the last one. A warning is raised if the shape leaves the
          choice ambiguous, which happens when the model returns as many
          quantiles as there are forecast steps. All the axes that are neither
          the series axis nor the horizon axis are taken to hold quantile levels
          or sample paths, and are reduced according to ``quantile``.
        quantile : float or None (default: 0.5)
          Probabilistic models return several quantiles or sample paths; this
          is the quantile used to derive a point forecast from them. When None,
          the mean is used instead. Note that the quantile is taken over the
          values the model returned, without assuming which level each of them
          corresponds to, since that information is not exposed in any
          standard way. For a model returning evenly spread quantile levels,
          the default therefore recovers the median forecast, up to any
          quantile crossing.
        model_kwargs : dict or None (default: None)
          Extra keyword arguments passed to the model's forecasting method.
        warn_on_fit : bool (default: True)
          Whether calling :meth:`fit` should emit a warning reminding that
          nothing is being learnt.

    Attributes
    ----------
        n_features_in_ : int
          Number of features (channels) of the series seen during fit.
        context_format_ : str
          How the batch of contexts ended up being presented to the model,
          among ``"2d"``, ``"3d"`` and ``"list"``. Set after the first call to
          :meth:`predict` with the ``"univariate"`` layout, as implementations
          disagree on this and the accepted one is discovered by trial.

    Notes
    -----
        Unlike :class:`~tslearn.forecasting.VARIMA`, this estimator holds no
        state about the fitted data, so :meth:`predict` requires ``X``.

    See Also
    --------
        LinearProbeForecaster: Fit a linear head on top of a frozen pre-trained model.

    Examples
    --------
    >>> from chronos import Chronos2Pipeline  # doctest: +SKIP
    >>> pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2")  # doctest: +SKIP
    >>> model = ZeroShotForecaster(pipeline)  # doctest: +SKIP
    >>> model.predict(X, n=24).shape  # doctest: +SKIP
    (10, 24, 1)

    References
    ----------
    .. [1] A. F. Ansari, L. Stella, C. Turkmen, et al. Chronos: Learning the
      Language of Time Series. Transactions on Machine Learning Research, 2024.

    """

    def __init__(
        self,
        model,
        predict_fn=None,
        input_layout="univariate",
        context_length=None,
        horizon_axis="auto",
        quantile=0.5,
        model_kwargs=None,
        warn_on_fit=True,
    ):
        self.model = model
        self.predict_fn = predict_fn
        self.input_layout = input_layout
        self.context_length = context_length
        self.horizon_axis = horizon_axis
        self.quantile = quantile
        self.model_kwargs = model_kwargs
        self.warn_on_fit = warn_on_fit

    def _resolve_predict_fn(self):
        """Build a ``(context, horizon) -> forecast`` callable from the model."""
        if self.predict_fn is not None:
            return lambda context, horizon: self.predict_fn(
                self.model, context, horizon
            )

        for method_name in CANDIDATE_PREDICT_METHODS:
            method = getattr(self.model, method_name, None)
            if not callable(method):
                continue
            try:
                parameters = inspect.signature(method).parameters
            except (TypeError, ValueError):  # pragma: no cover
                continue
            horizon_name = next(
                (name for name in CANDIDATE_HORIZON_NAMES if name in parameters),
                None,
            )
            if horizon_name is None:
                continue

            def call(context, horizon, method=method, horizon_name=horizon_name):
                kwargs = dict(self.model_kwargs or {})
                kwargs[horizon_name] = horizon
                return method(context, **kwargs)

            return call

        raise ValueError(
            "Could not find a forecasting method on the provided model. The "
            f"model is expected to expose one of {CANDIDATE_PREDICT_METHODS} "
            "accepting a forecast horizon. Pass an explicit `predict_fn` "
            "otherwise."
        )

    def _call_univariate(self, call, context, horizon):
        """Call the model, negotiating how the context batch is presented.

        Implementations disagree on whether a batch of univariate series should
        be handed over as a 2d array, as a 3d array with an explicit variate
        axis, or as a list of series. Rather than requiring the user to know,
        the candidate representations are tried in turn and the first one the
        model accepts is remembered for subsequent calls.
        """
        if self.predict_fn is not None:
            return call(context, horizon)

        known_format = getattr(self, "context_format_", None)
        formats = (
            (known_format,) if known_format is not None else CONTEXT_FORMATS
        )
        errors = {}
        for context_format in formats:
            try:
                forecast = call(_format_context(context, context_format), horizon)
            except (ValueError, TypeError, IndexError, RuntimeError) as error:
                errors[context_format] = f"{type(error).__name__}: {error}"
                continue
            self.context_format_ = context_format
            return forecast

        details = "\n".join(f"  - as a {key} input: {value}" for key, value in errors.items())
        raise ValueError(
            "The model rejected every supported way of passing a batch of "
            f"univariate contexts:\n{details}\nPass an explicit `predict_fn` to "
            "control how the model is called."
        )

    def fit(self, X, y=None):
        """Do nothing, as a zero-shot model needs no training.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.
            y : Ignored

        Returns
        -------
            self
                The estimator, ready to be used for prediction

        """
        self._check_layout()
        X = self._check_input(X)
        if self.warn_on_fit:
            warnings.warn(
                "ZeroShotForecaster.fit does not train anything: the wrapped "
                "model is used as-is for zero-shot forecasting. Use "
                "LinearProbeForecaster to fit a head on top of it, or pass "
                "warn_on_fit=False to silence this warning.",
                UserWarning,
                stacklevel=2,
            )
        self.n_features_in_ = X.shape[2]
        self._is_fitted_ = True
        return self

    def predict(self, X, n=1):
        """Forecast ``n`` timestamps ahead of the given series.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
              Time series dataset to forecast.
            n : int (default: 1)
              The number of timestamps to forecast, a.k.a. the horizon.

        Returns
        -------
            array of shape=(n_ts, n, d)
              Array of forecasted timestamps

        """
        self._check_layout()
        if X is None:
            raise ValueError(
                "A zero-shot model keeps no state about the data seen at fit "
                "time, so `X` is required at predict time."
            )
        if n < 1:
            raise ValueError(f"`n` must be a positive integer, got {n}.")
        X = self._check_input(X)
        if getattr(self, "n_features_in_", X.shape[2]) != X.shape[2]:
            raise ValueError(
                f"Series with {self.n_features_in_} features were expected, got "
                f"{X.shape[2]}."
            )
        if self.context_length is not None:
            X = X[:, -self.context_length :]

        n_ts, sz, d = X.shape
        context = _layout_to_model_input(X, self.input_layout)
        call = self._resolve_predict_fn()

        if self.input_layout == "univariate":
            forecast = _unwrap_forecast(self._call_univariate(call, context, n))
            forecast = _reduce_to_point_forecast(
                forecast, n_ts * d, n, self.quantile, self.horizon_axis
            )
            # (n_ts * d, n) -> (n_ts, n, d)
            return np.swapaxes(forecast.reshape(n_ts, d, n), 1, 2)

        forecast = np.asarray(
            _unwrap_forecast(call(context, n)), dtype=np.float64
        )
        if forecast.shape[:1] != (n_ts,):
            raise ValueError(
                f"The model returned forecasts for {forecast.shape[0]} series "
                f"while {n_ts} were provided."
            )
        if self.input_layout == "channels_first":
            forecast = np.swapaxes(forecast, 1, 2)
        if forecast.ndim != 3 or forecast.shape[2] != d:
            raise ValueError(
                f"Expected forecasts of shape (n_ts, n, d) = ({n_ts}, {n}, {d}) "
                f"with input_layout='{self.input_layout}', got shape "
                f"{forecast.shape}. Pass an explicit `predict_fn` to control "
                "how the model is called."
            )
        return forecast[:, :n]


class LinearProbeForecaster(_BaseFoundationForecaster):
    """Forecast by fitting a linear head on a frozen pre-trained model.

    Linear probing keeps the pre-trained model entirely frozen and only fits a
    linear map from its representations to the quantity of interest. Compared
    to zero-shot use, it adapts the model to the data at hand at a very small
    computational cost, and compared to fine-tuning, it leaves the pre-trained
    weights untouched, which makes it both cheap and hard to overfit.

    Training pairs are cut out of the provided series with a sliding window:
    the ``context_length`` timestamps preceding a cut point are embedded, and
    the ``horizon`` timestamps following it are used as the target.

    Parameters
    ----------
        model : torch.nn.Module
          A pre-trained model used as a frozen feature extractor.
        probe : sklearn estimator or None (default: None)
          The head fitted on top of the frozen representations. When None, a
          :class:`sklearn.linear_model.RidgeCV` is used, which selects its
          regularization strength by cross-validation and admits a closed-form
          solution. Any multi-output capable regressor can be passed instead;
          note that using a non-linear one makes the name "probe" a misnomer,
          and the resulting scores no longer measure linear separability of the
          representations. Frozen representations sometimes have a very small
          variance, in which case prepending a
          :class:`sklearn.preprocessing.StandardScaler` to the head helps.
        context_length : int (default: 128)
          Number of timestamps fed to the pre-trained model.
        horizon : int (default: 1)
          Number of timestamps forecasted at once. This is fixed at fit time,
          as one linear head is fitted for the whole horizon.
        stride : int (default: 1)
          Step between two consecutive training windows. Larger values yield
          fewer, less redundant training pairs and a faster fit.
        layer : int or None (default: None)
          Layer to probe, see
          :class:`~tslearn.foundation.TimeSeriesFoundationEmbedder`.
        layers_path : str or None (default: None)
          Dotted path to the stack of layers, see
          :class:`~tslearn.foundation.TimeSeriesFoundationEmbedder`.
        pooling : {"mean", "max", "cls", "last", "flatten"} (default: "mean")
          How token representations are aggregated, see
          :class:`~tslearn.foundation.TimeSeriesFoundationEmbedder`. Unlike
          that class, a probe needs a flat feature matrix, so ``pooling=None``
          is not accepted here.
        tokens : slice, (start, stop) pair or None (default: None)
          Which tokens to keep before pooling, see
          :class:`~tslearn.foundation.TimeSeriesFoundationEmbedder`. Restricting
          the average to the tokens that actually represent the context, e.g.
          ``tokens=(0, -2)`` for Chronos-2, avoids diluting it with class or
          forecast tokens.
        cls_index : int (default: 0)
          Index of the token selected when ``pooling="cls"``.
        input_layout : {"univariate", "channels_last", "channels_first"} (default: "univariate")
          How the context is laid out when handed over to the model.
        input_name : str or None (default: None)
          Name of the ``forward`` argument receiving the context values.
        model_kwargs : dict or None (default: None)
          Extra keyword arguments passed to every ``forward`` call.
        batch_size : int (default: 32)
          Number of series embedded at once.
        device : str or None (default: None)
          Device on which inference is run.
        verbose : int (default: 0)
          When positive, prints progress information.

    Attributes
    ----------
        embedder_ : TimeSeriesFoundationEmbedder
          The frozen feature extractor.
        probe_ : sklearn estimator
          The fitted linear head.
        n_features_in_ : int
          Number of features (channels) of the series seen during fit.
        n_windows_ : int
          Number of training windows cut out of the fit data.

    See Also
    --------
        ZeroShotForecaster: Use a pre-trained model without any training.
        LinearProbeClassifier: Linear probing for time series classification.

    Examples
    --------
    >>> model = LinearProbeForecaster(backbone, horizon=24)  # doctest: +SKIP
    >>> model.fit(X_train).predict(X_test, n=24).shape  # doctest: +SKIP
    (10, 24, 1)

    References
    ----------
    .. [1] G. Alain and Y. Bengio. Understanding intermediate layers using
      linear classifier probes. ICLR Workshop, 2017.

    """

    def __init__(
        self,
        model,
        probe=None,
        context_length=128,
        horizon=1,
        stride=1,
        layer=None,
        layers_path=None,
        pooling="mean",
        tokens=None,
        cls_index=0,
        input_layout="univariate",
        input_name=None,
        model_kwargs=None,
        batch_size=32,
        device=None,
        verbose=0,
    ):
        self.model = model
        self.probe = probe
        self.context_length = context_length
        self.horizon = horizon
        self.stride = stride
        self.layer = layer
        self.layers_path = layers_path
        self.pooling = pooling
        self.tokens = tokens
        self.cls_index = cls_index
        self.input_layout = input_layout
        self.input_name = input_name
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def _make_embedder(self):
        return TimeSeriesFoundationEmbedder(
            model=self.model,
            layer=self.layer,
            layers_path=self.layers_path,
            pooling=self.pooling,
            tokens=self.tokens,
            cls_index=self.cls_index,
            input_layout=self.input_layout,
            context_length=None,
            input_name=self.input_name,
            model_kwargs=self.model_kwargs,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
        )

    def _make_probe(self):
        if self.probe is None:
            return RidgeCV(alphas=np.logspace(-3, 3, 13))
        return clone(self.probe)

    def _make_windows(self, X):
        """Cut (context, target) pairs out of a (n_ts, sz, d) dataset."""
        n_ts, sz, d = X.shape
        span = self.context_length + self.horizon
        if sz < span:
            raise ValueError(
                f"Series of length at least context_length + horizon = {span} "
                f"are required to build training windows, got {sz}."
            )
        starts = np.arange(0, sz - span + 1, self.stride)
        contexts = np.stack(
            [X[:, start : start + self.context_length] for start in starts], axis=1
        )
        targets = np.stack(
            [
                X[:, start + self.context_length : start + span]
                for start in starts
            ],
            axis=1,
        )
        # (n_ts, n_windows, ...) -> (n_ts * n_windows, ...)
        contexts = contexts.reshape(n_ts * len(starts), self.context_length, d)
        targets = targets.reshape(n_ts * len(starts), self.horizon * d)
        return contexts, targets

    def _validate_params_(self):
        self._check_layout()
        _check_probe_pooling(self.pooling)
        for name in ("context_length", "horizon", "stride"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(
                    f"`{name}` must be a positive integer, got {value}."
                )

    def fit(self, X, y=None):
        """Fit a linear head on top of the frozen pre-trained model.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset, with ``sz >= context_length + horizon``.
            y : Ignored

        Returns
        -------
            self
                The fitted estimator

        """
        self._validate_params_()
        _require_torch()
        X = self._check_input(X)
        contexts, targets = self._make_windows(X)

        self.embedder_ = self._make_embedder()
        embeddings = self.embedder_.fit_transform(contexts)

        self.probe_ = self._make_probe()
        self.probe_.fit(embeddings, targets)

        self.n_features_in_ = X.shape[2]
        self.n_windows_ = contexts.shape[0]
        return self

    def predict(self, X, n=None):
        """Forecast ``horizon`` timestamps ahead of the given series.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
              Time series dataset to forecast, with
              ``sz >= context_length``.
            n : int or None (default: None)
              The number of timestamps to forecast. A single head is fitted for
              the whole horizon, so ``n`` may not exceed the ``horizon`` value
              used at fit time. When None, ``horizon`` is used.

        Returns
        -------
            array of shape=(n_ts, n, d)
              Array of forecasted timestamps

        """
        check_is_fitted(self, "probe_")
        n = self.horizon if n is None else n
        if not 1 <= n <= self.horizon:
            raise ValueError(
                f"This estimator was fitted for a horizon of {self.horizon}, so "
                f"`n` must lie in [1, {self.horizon}], got {n}. Refit with a "
                "larger `horizon` to forecast further ahead."
            )
        X = self._check_input(X)
        if X.shape[2] != self.n_features_in_:
            raise ValueError(
                f"Series with {self.n_features_in_} features were expected, got "
                f"{X.shape[2]}."
            )
        if X.shape[1] < self.context_length:
            raise ValueError(
                f"Series of length at least context_length="
                f"{self.context_length} are required, got {X.shape[1]}."
            )
        contexts = X[:, -self.context_length :]
        predictions = self.probe_.predict(self.embedder_.transform(contexts))
        predictions = np.asarray(predictions, dtype=np.float64)
        predictions = predictions.reshape(X.shape[0], self.horizon, self.n_features_in_)
        return predictions[:, :n]

    def score(self, X, y=None):
        """Return the coefficient of determination of the forecast.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset, with ``sz >= context_length + horizon``.
            y : Ignored

        Returns
        -------
            float
              :math:`R^2` of the forecast over all windows cut out of ``X``.

        """
        check_is_fitted(self, "probe_")
        X = self._check_input(X)
        contexts, targets = self._make_windows(X)
        return self.probe_.score(self.embedder_.transform(contexts), targets)
