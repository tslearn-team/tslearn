"""
Forecasting with a pre-trained model
====================================

This example showcases the :mod:`tslearn.foundation` module, which allows one
to re-use a pre-trained time series model behind the usual tslearn API. Two
adaptation strategies are compared on the same data:

* zero-shot forecasting, with :class:`~tslearn.foundation.ZeroShotForecaster`,
  where the pre-trained model is used exactly as it is;
* linear probing, with :class:`~tslearn.foundation.LinearProbeForecaster`,
  where the pre-trained model is kept frozen and only a linear head mapping its
  representations to future values is fitted.

Both are compared to the classical :class:`~tslearn.forecasting.AutoVARIMA`
baseline.

The pre-trained model used here is Chronos-2 [1]_, an encoder-only model
pre-trained on a large corpus of real and synthetic series. Running this
example therefore requires the ``chronos-forecasting`` package::

    pip install "chronos-forecasting>=2.0"

Note that nothing in :mod:`tslearn.foundation` is specific to Chronos: any
PyTorch model exposing similar conventions can be plugged in the same way.

References
----------
.. [1] A. F. Ansari, O. Shchur, J. Küken, et al. Chronos-2: From Univariate to
  Universal Forecasting. arXiv:2510.15821, 2025.
"""

##############################################################################
# Data
# ----
#
# We use a set of sine waves with varying frequencies, phases and noise levels.

import numpy as np

from tslearn.preprocessing import TimeSeriesScalerMeanVariance

rng = np.random.RandomState(0)

n_ts, sz, horizon, context_length = 32, 512, 48, 128
t = np.arange(sz + horizon)

periods = rng.uniform(30, 50, size=n_ts)
phases = rng.uniform(0, 2 * np.pi, size=n_ts)
trends = rng.uniform(-2e-3, 2e-3, size=n_ts)

full_series = (
    np.sin(2 * np.pi * t[None, :] / periods[:, None] + phases[:, None])
    + trends[:, None] * t[None, :]
    + 0.1 * rng.randn(n_ts, sz + horizon)
)
full_series = TimeSeriesScalerMeanVariance().fit_transform(full_series)

X_train, X_test = full_series[:, :sz], full_series[:, sz:]
print(f"{X_train.shape=}, {X_test.shape=}")

##############################################################################
# Zero-shot forecasting
# ---------------------
#
# A time series foundation model is pre-trained to forecast unseen series out
# of the box, so no training is needed at all. Wrapping the inference pipeline
# in a :class:`~tslearn.foundation.ZeroShotForecaster` is enough to obtain a
# tslearn estimator, whose ``predict`` method returns an array of shape
# ``(n_ts, n, d)`` like every other forecaster of the library.
#
# Calling ``fit`` is not needed here since the purpose of a 
# :class:`~tslearn.foundation.ZeroShotForecaster` is to reuse a pre-trained 
# model without any training.

from chronos import Chronos2Pipeline

from tslearn.foundation import ZeroShotForecaster

pipeline = Chronos2Pipeline.from_pretrained("autogluon/chronos-2-small")

zero_shot = ZeroShotForecaster(pipeline, horizon_axis=-1)
y_zero_shot = zero_shot.predict(X_train, n=horizon)
print(f"{y_zero_shot.shape=}")

##############################################################################
# ``horizon_axis`` says which axis of the model's output holds the forecast
# horizon. There is no shared convention across implementations, Chronos-2
# returns ``(n_series, n_quantiles, horizon)`` from ``predict`` so ``-1`` is 
# used. Left to its default of ``"auto"``, the estimator would
# infer it from the returned shape.
#
# The remaining axes are understood as holding quantile levels or sample paths
# and are reduced to a point forecast according to the ``quantile`` parameter,
# which defaults to the median.

##############################################################################
# Linear probing
# --------------
#
# Zero-shot use ignores the fact that we do have training data at hand. Linear
# probing exploits it, while leaving the pre-trained weights untouched: the
# model is only used to embed the context window, and a linear map from those
# representations to the next ``horizon`` values is fitted.
#
# Training pairs are cut out of the series with a sliding window. A larger
# ``stride`` yields fewer, less redundant windows and a faster fit, which
# matters because every window has to go through the pre-trained model once.
#
# The probing estimator needs the model itself rather than the inference
# pipeline, since it reads hidden states rather than forecasts. Three options
# drive which representations are used:
#
# * ``layer`` selects the block to probe: an integer places a forward hook on
#   the corresponding block, so ``layer=-2`` probes the penultimate one. The
#   last layers of a pre-trained model tend to specialize towards its
#   pre-training objective, so probing slightly earlier is often beneficial.
#   ``layer=None`` reads the model's own output instead, which only works for
#   models returning their hidden states; Chronos-2 returns quantile forecasts
#   only, so an explicit layer is required here.
# * ``pooling`` says how the token representations are aggregated into a single
#   vector. Chronos-2 emits one token per patch of the context, plus a register
#   token and a forecast token, so both averaging (``pooling="mean"``) and
#   picking a single token (``pooling="token"``) are sensible. 
#   Chronos-2's forecast token is a natural choice
#   here: ``token_index=-1`` reads it directly, rather than averaging over
#   representations that were never meant to summarize the series for
#   forecasting.
# * ``tokens`` restricts which tokens take part in that aggregation. Chronos-2
#   appends a register token and a forecast token after its context tokens, and
#   neither represents the input series, so ``tokens=(0, -2)`` focuses on the 
#   slice 0:-2 and keeps the average clean.
#
# .. image:: /_static/img/foundation_tokens.svg
#    :width: 700
#    :align: center
#    :alt: Chronos-2 appends a register token and a forecast token after its
#      context tokens; tokens=(0, -2) keeps only the context tokens.

from tslearn.foundation import LinearProbeForecaster

probe = LinearProbeForecaster(
    pipeline.model,
    context_length=context_length,
    horizon=horizon,
    stride=16,
    layer=-2,
    pooling="mean",
    tokens=(0, -2),
)
probe.fit(X_train)
y_probe = probe.predict(X_train)

print(f"{probe.n_windows_} training windows, "
      f"{probe.embedder_.embedding_size_}-dimensional embeddings")

##############################################################################
# The head fitted on top of the frozen representations defaults to a
# :class:`sklearn.linear_model.RidgeCV`, which picks its regularization
# strength by cross-validation. Any scikit-learn regressor can be passed
# instead through the ``probe`` parameter.

##############################################################################
# Comparison
# ----------
#
# Both strategies are compared to AutoVARIMA on the mean absolute error over
# the held-out horizon.

from tslearn.forecasting import AutoVARIMA
from tslearn.metrics.performance import mae

varima = AutoVARIMA(seasonal_period=40).fit(X_train)
y_varima = varima.predict(n=horizon)

scores = {
    "AutoVARIMA (with seasonal differentiation)": mae(X_test, y_varima),
    "Chronos-2 (zero-shot)": mae(X_test, y_zero_shot),
    "Chronos-2 (linear probe)": mae(X_test, y_probe),
}
for name, score in scores.items():
    print(f"{name:>42}: MAE = {score:.4f}")

##############################################################################
# Let us look at a few series to see where the differences come from.

import matplotlib.pyplot as plt

shown = [0, 1, 2]
fig, axes = plt.subplots(
    len(shown), 1, sharex=True, figsize=(11, 8), layout="constrained"
)
context = 150
for ax, i in zip(axes, shown):
    ax.plot(
        np.arange(sz - context, sz),
        X_train[i, -context:, 0],
        color="0.4",
        label="context",
    )
    ax.plot(
        np.arange(sz, sz + horizon),
        X_test[i, :, 0],
        color="k",
        label="ground truth",
    )
    for values, label in [
        (y_varima, "AutoVARIMA"),
        (y_zero_shot, "zero-shot"),
        (y_probe, "linear probe"),
    ]:
        ax.plot(np.arange(sz, sz + horizon), values[i, :, 0], label=label, alpha=0.8)
    ax.axvline(sz, color="0.8", linestyle="--")
    ax.set_ylabel(f"series {i}")
axes[0].legend(loc="upper left", ncol=5, fontsize="small")
fig.suptitle("Forecasting a held-out horizon", fontsize=14)
plt.show()

##############################################################################
# Choosing what to probe
# ----------------------
#
# Which layer and which pooling work best is data-dependent, and cheap enough
# to explore: the frozen model is the expensive part, and it is never trained.
# Below we score a few configurations on the held-out horizon.

results = {}
for layer in [-1, -2, -4]:
    for pooling in ["mean", "token"]:
        model = LinearProbeForecaster(
            pipeline.model,
            context_length=context_length,
            horizon=horizon,
            stride=48,
            layer=layer,
            pooling=pooling,
            # Chronos-2's forecast token is its last one
            token_index=-1,
            tokens=(0, -2),
        ).fit(X_train)
        results[(layer, pooling)] = mae(X_test, model.predict(X_train))

for (layer, pooling), score in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"layer={str(layer):>5}, pooling={pooling:>4}: MAE = {score:.4f}")
