"""
The :mod:`tslearn.foundation` module gathers estimators that re-use pre-trained
time series models, such as the ones published on the Hugging Face Hub, behind
the usual tslearn API.

Three adaptation strategies are covered:

* zero-shot forecasting, with :class:`ZeroShotForecaster`, which uses a
  pre-trained forecaster as-is;
* linear probing for forecasting, with :class:`LinearProbeForecaster`;
* linear probing for classification, with :class:`LinearProbeClassifier`.

The last two rely on :class:`TimeSeriesFoundationEmbedder`, which turns any
frozen PyTorch model into a feature extractor and lets one choose which layer
to read representations from and how to pool them.

These estimators are deliberately agnostic to any specific model
implementation: they duck-type the wrapped object and offer escape hatches
(``predict_fn``, ``input_name``, ``layers_path``, ``input_layout``) for models
that depart from the most widespread conventions.

Notes
-----
    This module requires PyTorch, which is an optional dependency of tslearn.

"""

from ._classification import LinearProbeClassifier
from ._embedding import TimeSeriesFoundationEmbedder
from ._forecasting import LinearProbeForecaster, ZeroShotForecaster

__all__ = [
    "LinearProbeClassifier",
    "LinearProbeForecaster",
    "TimeSeriesFoundationEmbedder",
    "ZeroShotForecaster",
]
