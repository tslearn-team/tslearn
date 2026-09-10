"""
The :mod:`tslearn.foundation` module gathers estimators that re-use pre-trained
time series models, such as the ones published on the Hugging Face Hub, behind
the usual tslearn API.

Two adaptation strategies are covered:

* zero-shot forecasting, with :class:`ZeroShotForecaster`, which uses a
  pre-trained forecaster as-is;
* linear probing for forecasting, with :class:`LinearProbeForecaster`.

Linear probing relies on :class:`TimeSeriesFoundationEmbedder`, which turns any frozen
PyTorch model into a feature extractor and lets one choose which layer to
read representations from and how to pool them. Being a regular
scikit-learn transformer, it also covers linear probing for classification,
or any other downstream task, by composing with a
:class:`sklearn.pipeline.Pipeline`.

These estimators are deliberately agnostic to any specific model
implementation: they duck-type the wrapped object and offer escape hatches
(``predict_fn``, ``input_name``, ``layers_path``, ``input_layout``) for models
that depart from the most widespread conventions.

Notes
-----
    This module requires PyTorch, which is an optional dependency of tslearn.

"""

from ._embedding import TimeSeriesFoundationEmbedder
from ._forecasting import LinearProbeForecaster, ZeroShotForecaster

__all__ = [
    "LinearProbeForecaster",
    "TimeSeriesFoundationEmbedder",
    "ZeroShotForecaster",
]
