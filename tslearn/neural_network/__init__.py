"""
The :mod:`tslearn.neural_network` module contains multi-layer perceptron
models for time series classification and regression.

These are straight-forward adaptations of scikit-learn models.
"""

from .neural_network import TimeSeriesMLPClassifier, TimeSeriesMLPRegressor

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


__all__ = [
    "TimeSeriesMLPClassifier", "TimeSeriesMLPRegressor"
]
