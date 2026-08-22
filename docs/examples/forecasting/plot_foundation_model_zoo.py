"""
A tour of the pre-trained forecasting model zoo
=================================================

:mod:`tslearn.foundation` does not target a specific pre-trained model: it
only assumes that a model exposes a forecasting method, or a PyTorch
``forward``, following one of a handful of widespread conventions (see
:class:`~tslearn.foundation.ZeroShotForecaster` and
:class:`~tslearn.foundation.LinearProbeForecaster`). This example surveys
seven models for time series
forecasting, and shows, for each of them, what it takes to plug it in as a
:class:`~tslearn.foundation.ZeroShotForecaster`, a
:class:`~tslearn.foundation.LinearProbeForecaster`, or both. The smallest
checkpoint published for every model is used throughout, to keep downloads
light.

For a line-by-line walkthrough of the module's parameters, see the
:doc:`Chronos-2 example <plot_foundation_forecasting>`, which this one
complements rather than repeats.

=================  =======================  =========  ============
Model              Package                  Zero-shot  Linear probe
=================  =======================  =========  ============
Chronos-2 [1]_     ``chronos-forecasting``  yes        yes
Chronos-Bolt [2]_  ``chronos-forecasting``  yes        yes
TimesFM [3]_       ``timesfm``              yes        --
Moirai [4]_        ``uni2ts``               yes        --
TTM [5]_           ``granite-tsfm``         yes        --
MOMENT [6]_        ``momentfm``             --         yes
Time-MoE [7]_      ``transformers``         yes        yes
=================  =======================  =========  ============

The right-hand column is not a limitation of tslearn: it reflects how each
model exposes its internals. Models built around a clean
``forward(context) -> hidden_states`` (the two Chronos variants, Time-MoE)
probe as easily as they forecast zero-shot. Models whose low-level module
expects an already-patchified input, an explicit padding mask, or a packed
multi-series format (TimesFM, Moirai, TTM) are naturally used zero-shot, and
would need a bespoke wrapper to expose clean per-token representations for
probing. MOMENT is the mirror case: only its reconstruction head is
pre-trained, so its zero-shot *forecasts* would come out of a head that has
never been trained, while its embeddings, precisely what linear probing
needs, are excellent.

References
----------
.. [1] A. F. Ansari, O. Shchur, J. Küken, et al. Chronos-2: From Univariate
  to Universal Forecasting. arXiv:2510.15821, 2025.
.. [2] A. F. Ansari, L. Stella, C. Turkmen, et al. Chronos: Learning the
  Language of Time Series. TMLR, 2024.
.. [3] A. Das, W. Kong, R. Sen, Y. Zhou. A decoder-only foundation model for
  time-series forecasting. ICML, 2024.
.. [4] G. Woo, C. Liu, A. Kumar, et al. Unified Training of Universal Time
  Series Forecasting Transformers. ICML, 2024.
.. [5] V. Ekambaram, A. Jati, P. Dayama, et al. Tiny Time Mixers (TTMs): Fast
  Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate
  Time Series. NeurIPS, 2024.
.. [6] M. Goswami, K. Szafer, A. Choudhry, et al. MOMENT: A Family of Open
  Time-series Foundation Models. ICML, 2024.
.. [7] X. Shi, S. Wang, Y. Nie, et al. Time-MoE: Billion-Scale Time Series
  Foundation Models with Mixture of Experts. ICLR, 2025.
"""

##############################################################################
# Data
# ----
#
# The same small set of noisy sine waves used by the Chronos-2 example is
# reused here, so that Chronos-Bolt's forecasts below can be read against
# the same yardstick.

import numpy as np

from tslearn.utils import to_time_series_dataset

rng = np.random.RandomState(0)

n_ts, sz, horizon, context_length = 16, 256, 24, 96
t = np.arange(sz + horizon)

periods = rng.uniform(30, 50, size=n_ts)
phases = rng.uniform(0, 2 * np.pi, size=n_ts)

full_series = np.sin(
    2 * np.pi * t[None, :] / periods[:, None] + phases[:, None]
) + 0.1 * rng.randn(n_ts, sz + horizon)
full_series = to_time_series_dataset(full_series)

X_train, X_test = full_series[:, :sz], full_series[:, sz:]

##############################################################################
# Chronos-2
# ---------
#
# Chronos-2 [1]_ is covered in depth by the :doc:`plot_foundation_forecasting`
# example: zero-shot forecasting with :class:`~tslearn.foundation.ZeroShotForecaster`
# needs nothing beyond the pipeline itself, and linear probing with
# :class:`~tslearn.foundation.LinearProbeForecaster` reads its forecast token
# (``layer=-2, pooling="token", token_index=-1, tokens=(0, -2)``). It is not
# repeated here.

##############################################################################
# Chronos-Bolt
# ------------
#
# Chronos-Bolt [2]_ is a faster, encoder-decoder T5 model, distilled to
# predict all quantiles in a single forward pass rather than
# autoregressively, from the same package as Chronos-2::
#
#     pip install "chronos-forecasting>=2.0"
#
# Two details set it apart from Chronos-2:
#
# * its ``predict`` method requires a :class:`torch.Tensor`, not a NumPy
#   array, so the auto-detected calling convention of
#   :class:`~tslearn.foundation.ZeroShotForecaster` does not apply and an
#   explicit ``predict_fn`` is needed;
# * its ``forward`` takes a raw ``context`` and exposes an encoder made of a
#   plain stack of T5 blocks, at ``encoder.block``, with no register or
#   forecast token to filter out, so probing needs no ``tokens`` argument.
#
# As with Chronos-2, the raw series are passed to
# :class:`~tslearn.foundation.ZeroShotForecaster` unscaled, since it wraps
# the full pipeline. The linear probe, on the other hand, reads
# ``pipeline.model`` directly and so needs scaling restored explicitly through
# a :class:`~sklearn.pipeline.Pipeline`, as detailed in the
# :doc:`plot_foundation_forecasting` example.

import torch
from chronos import BaseChronosPipeline
from sklearn.pipeline import Pipeline

from tslearn.foundation import LinearProbeForecaster, ZeroShotForecaster
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small", device_map="cpu"
)

zero_shot = ZeroShotForecaster(
    pipeline,
    predict_fn=lambda model, context, horizon: model.predict(
        torch.as_tensor(context, dtype=torch.float32), prediction_length=horizon
    ),
    horizon_axis=-1,
)
y_zero_shot = zero_shot.predict(X_train, n=horizon)

forecaster = LinearProbeForecaster(
    pipeline.model,
    context_length=context_length,
    horizon=horizon,
    stride=8,
    layer=-1,
    layers_path="encoder.block",
    pooling="mean",
)
scaler = TimeSeriesScalerMeanVariance(per_timeseries=False)
probe = Pipeline([("scale", scaler), ("probe", forecaster)])
probe.fit(X_train)

# The scaler also normalizes the training targets, so forecasts come out on
# that scale and are put back in the original units, using the ``mean_`` and
# ``std_`` the scaler computed when it was fitted, before comparison.
y_probe = probe.predict(X_train) * scaler.std_ + scaler.mean_

from tslearn.metrics.performance import mae

print(f"{'Chronos-Bolt (zero-shot)':>28}: MAE = {mae(X_test, y_zero_shot):.4f}")
print(f"{'Chronos-Bolt (linear probe)':>28}: MAE = {mae(X_test, y_probe):.4f}")

##############################################################################
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 3), layout="constrained")
context = 80
ax.plot(np.arange(sz - context, sz), X_train[0, -context:, 0], color="0.4", label="context")
ax.plot(np.arange(sz, sz + horizon), X_test[0, :, 0], color="k", label="ground truth")
ax.plot(np.arange(sz, sz + horizon), y_zero_shot[0, :, 0], label="zero-shot", alpha=0.8)
ax.plot(np.arange(sz, sz + horizon), y_probe[0, :, 0], label="linear probe", alpha=0.8)
ax.axvline(sz, color="0.8", linestyle="--")
ax.legend(loc="upper left", ncol=4, fontsize="small")
ax.set_title("Chronos-Bolt")
plt.show()

##############################################################################
# TimesFM
# -------
#
# TimesFM [3]_ is a decoder-only model, pre-trained to predict a whole patch
# of future values at once rather than one step at a time::
#
#     pip install "timesfm[torch]"
#
# Its ``forecast`` method takes the horizon *before* the context
# (``forecast(horizon, inputs)``), which conflicts with the positional
# calling convention :class:`~tslearn.foundation.ZeroShotForecaster` uses
# when auto-detecting a method, so ``predict_fn`` is required here too. The
# horizon and context length are fixed once and for all through a
# :class:`~timesfm.ForecastConfig` passed to ``compile``, which builds the
# model's static computation graph.
#
# .. code-block:: python
#
#     import timesfm
#
#     from tslearn.foundation import ZeroShotForecaster
#
#     model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
#         "google/timesfm-2.5-200m-pytorch"
#     )
#     model.compile(timesfm.ForecastConfig(
#         max_context=context_length, max_horizon=horizon, normalize_inputs=True
#     ))
#
#     zero_shot = ZeroShotForecaster(
#         model,
#         predict_fn=lambda model, context, horizon: model.forecast(
#             horizon=horizon, inputs=list(context)
#         )[0],
#         context_length=context_length,
#     )
#     y_zero_shot = zero_shot.predict(X_train, n=horizon)
#
# TimesFM's own module (``model.model``) does not lend itself to linear
# probing out of the box: its ``forward`` expects the context already split
# into patches and concatenated with an explicit padding mask, a
# transformation that :class:`~tslearn.foundation.TimeSeriesFoundationEmbedder`
# does not perform, so probing it would require wrapping that patching logic
# in a small ``torch.nn.Module`` of one's own.

##############################################################################
# Moirai
# ------
#
# Moirai [4]_ is a masked-encoder model, natively multivariate and trained
# across many frequencies at once::
#
#     pip install "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts.git"
#
# The forecasting horizon, and the number of sample paths drawn from the
# predictive distribution, are baked into the model at construction time
# rather than passed to a ``predict``-like method, so, again, a
# ``predict_fn`` builds the ``past_target`` / ``past_observed_target`` /
# ``past_is_pad`` triplet that :class:`~uni2ts.model.moirai.MoiraiForecast`
# expects.
#
# .. code-block:: python
#
#     import torch
#
#     from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
#
#     from tslearn.foundation import ZeroShotForecaster
#
#     module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
#     forecast_model = MoiraiForecast(
#         module=module,
#         prediction_length=horizon,
#         context_length=context_length,
#         patch_size=32,
#         target_dim=1,
#         feat_dynamic_real_dim=0,
#         past_feat_dynamic_real_dim=0,
#     )
#
#     def predict_fn(model, context, horizon):
#         past_target = torch.as_tensor(context, dtype=torch.float32).unsqueeze(-1)
#         past_observed = torch.ones_like(past_target, dtype=torch.bool)
#         past_is_pad = torch.zeros(past_target.shape[:2], dtype=torch.bool)
#         return model(past_target, past_observed, past_is_pad, num_samples=20)
#
#     zero_shot = ZeroShotForecaster(
#         forecast_model, predict_fn=predict_fn, context_length=context_length
#     )
#     y_zero_shot = zero_shot.predict(X_train, n=horizon)
#
# Its backbone (``MoiraiModule``) consumes a packed representation of the
# whole batch, with explicit ``sample_id``, ``variate_id`` and ``time_id``
# tensors describing which timestep and which series each row belongs to.
# Reproducing that packing outside of ``uni2ts`` itself is enough work that
# zero-shot use is, in practice, the natural way to reach for Moirai.

##############################################################################
# TTM (Tiny Time Mixers)
# -----------------------
#
# TTM [5]_ departs from the transformer architecture used by every other
# model in this list: it is a light-weight MLP-Mixer, which is also why it
# is small enough to not need a dedicated "small" checkpoint::
#
#     pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.22"
#
# Its context length and horizon are fixed by the checkpoint (512 and 96 for
# ``granite-timeseries-ttm-r2``) rather than adjustable at call time, and,
# since the raw model is called directly, ``predict_fn`` again does the
# work that a ``predict`` method would otherwise do.
#
# .. code-block:: python
#
#     import torch
#
#     from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
#
#     from tslearn.foundation import ZeroShotForecaster
#
#     model = TinyTimeMixerForPrediction.from_pretrained(
#         "ibm-granite/granite-timeseries-ttm-r2", num_input_channels=1
#     )
#
#     def predict_fn(model, context, horizon):
#         past_values = torch.as_tensor(context, dtype=torch.float32).unsqueeze(-1)
#         return model(past_values=past_values).prediction_outputs
#
#     zero_shot = ZeroShotForecaster(
#         model, predict_fn=predict_fn, context_length=model.config.context_length
#     )
#     y_zero_shot = zero_shot.predict(X_train, n=model.config.prediction_length)
#
# TTM's mixer blocks operate on a ``(batch, channels, patches, dim)`` tensor
# rather than the ``(batch, tokens, dim)`` shape
# :class:`~tslearn.foundation.TimeSeriesFoundationEmbedder` expects, so
# probing it directly would need a small reshaping wrapper too.

##############################################################################
# MOMENT
# ------
#
# MOMENT [6]_ is pre-trained as a masked autoencoder: reconstructing masked
# patches is what gives it strong general-purpose embeddings, but it means
# its forecasting head, unlike its embeddings, has never actually been
# trained::
#
#     pip install momentfm
#
# Loaded with ``task_name="embedding"``, a forward pass already returns a
# single pooled vector per series (``output.embeddings``), so this is the
# rare model that needs no ``layer``/``pooling`` gymnastics to reach a
# feature matrix, only a hook on its encoder to read the token-level
# representations that :class:`~tslearn.foundation.LinearProbeForecaster`
# expects.
#
# As with the linear probes above, ``model`` is the bare backbone rather than
# a pipeline, so scaling has to be restored the same way, through a
# :class:`~sklearn.pipeline.Pipeline`.
#
# .. code-block:: python
#
#     from momentfm import MOMENTPipeline
#     from sklearn.pipeline import Pipeline
#
#     from tslearn.foundation import LinearProbeForecaster
#     from tslearn.preprocessing import TimeSeriesScalerMeanVariance
#
#     model = MOMENTPipeline.from_pretrained(
#         "AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"}
#     )
#     model.init()
#
#     scaler = TimeSeriesScalerMeanVariance(per_timeseries=False)
#     probe = Pipeline([("scale", scaler), ("probe", LinearProbeForecaster(
#         model,
#         context_length=512,
#         horizon=horizon,
#         stride=32,
#         layer=-1,
#         layers_path="encoder.block",
#         pooling="mean",
#         input_layout="channels_first",
#     ))])
#     probe.fit(X_train)
#     y_probe = probe.predict(X_train) * scaler.std_ + scaler.mean_
#
# ``input_layout="channels_first"`` matters here: MOMENT's ``forward``
# expects an explicit channel axis (``x_enc`` of shape
# ``(batch, channels, sz)``), unlike most other models in this list which
# take a plain ``(batch, sz)`` batch of univariate contexts. Calling
# :meth:`~tslearn.foundation.LinearProbeForecaster.predict` (a zero-shot
# forecast) would technically run, but on a randomly initialized head, so it
# is deliberately left out.

##############################################################################
# Time-MoE
# --------
#
# Time-MoE [7]_ is a decoder-only, GPT-style model trained with next-value
# prediction and a sparse mixture-of-experts feed-forward block, distributed
# as ``transformers`` "remote code" rather than through its own package::
#
#     pip install transformers
#
# .. code-block:: python
#
#     from transformers import AutoModelForCausalLM
#
#     from tslearn.foundation import LinearProbeForecaster, ZeroShotForecaster
#
#     model = AutoModelForCausalLM.from_pretrained(
#         "Maple728/TimeMoE-50M", trust_remote_code=True
#     )
#
# Forecasting several steps ahead means generating token by token, through
# ``generate`` rather than a single forward pass; ``max_new_tokens`` is not
# one of the horizon argument names
# :class:`~tslearn.foundation.ZeroShotForecaster` looks for, and the output
# needs slicing to drop the echoed context, so a ``predict_fn`` is used once
# more:
#
# .. code-block:: python
#
#     def predict_fn(model, context, horizon):
#         context = torch.as_tensor(context, dtype=torch.float32)
#         out = model.generate(input_ids=context, max_new_tokens=horizon)
#         return out[:, -horizon:]
#
#     zero_shot = ZeroShotForecaster(model, predict_fn=predict_fn)
#     y_zero_shot = zero_shot.predict(X_train, n=horizon)
#
# Being decoder-only, its natural pooling is ``"last"``, the representation
# of the final context token, which has attended to every earlier one. Here
# too, ``model`` is the bare backbone rather than a normalizing pipeline, so
# scaling is restored through a :class:`~sklearn.pipeline.Pipeline`:
#
# .. code-block:: python
#
#     from sklearn.pipeline import Pipeline
#
#     from tslearn.preprocessing import TimeSeriesScalerMeanVariance
#
#     scaler = TimeSeriesScalerMeanVariance(per_timeseries=False)
#     probe = Pipeline([("scale", scaler), ("probe", LinearProbeForecaster(
#         model,
#         context_length=context_length,
#         horizon=horizon,
#         stride=8,
#         layer=-1,
#         layers_path="model.layers",
#         pooling="last",
#     ))])
#     probe.fit(X_train)
#     y_probe = probe.predict(X_train) * scaler.std_ + scaler.mean_
