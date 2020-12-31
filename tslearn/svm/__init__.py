"""
The :mod:`tslearn.svm` module contains Support Vector Classifier (SVC) and
Support Vector Regressor (SVR) models for time series.
"""

from .svm import TimeSeriesSVC, TimeSeriesSVR, TimeSeriesSVMMixin

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

__all__ = ["TimeSeriesSVC", "TimeSeriesSVR", "TimeSeriesSVMMixin"]
