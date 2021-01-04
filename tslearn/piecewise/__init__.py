"""
The :mod:`tslearn.piecewise` module gathers time series piecewise
approximation algorithms.
"""

from .piecewise import (PiecewiseAggregateApproximation,
                        SymbolicAggregateApproximation,
                        OneD_SymbolicAggregateApproximation)



__all__ = ["PiecewiseAggregateApproximation",
           "SymbolicAggregateApproximation",
           "OneD_SymbolicAggregateApproximation"]
