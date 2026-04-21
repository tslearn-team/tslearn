"""
The :mod:`tslearn.forecasting` module gathers time series specific forecasting
algorithms.

"""

from ._arima import VARIMA, AutoVARIMA

__all__ = ["VARIMA", "AutoVARIMA"]
