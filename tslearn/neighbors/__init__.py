"""
The :mod:`tslearn.neighbors` module gathers nearest neighbor algorithms using
time series metrics.
"""

from .neighbors import (
    KNeighborsTimeSeries,
    KNeighborsTimeSeriesClassifier,
    KNeighborsTimeSeriesRegressor
)

__all__ = [
    "KNeighborsTimeSeries",
    "KNeighborsTimeSeriesRegressor",
    "KNeighborsTimeSeriesClassifier"
]