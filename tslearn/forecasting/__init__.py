"""
The :mod:`tslearn.forecasting` module gathers time series specific forecasting
algorithms.

"""

from ._arima import VARIMA, AutoVARIMA
from ._pipeline import ScaledForecaster

__all__ = ["VARIMA", "AutoVARIMA", "ScaledForecaster"]
