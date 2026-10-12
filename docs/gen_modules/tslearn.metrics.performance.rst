.. _mod-metrics-performance:

tslearn.metrics.performance
===========================

The target and prediction datasets must have the same numbers of time series,
timestamps and features after conversion with
:func:`~tslearn.utils.to_time_series_dataset`. Having the same total number of
values is not sufficient: these metrics compare corresponding observations,
and raise ``ValueError`` when the dataset shapes differ.

.. automodule:: tslearn.metrics.performance

   .. rubric:: Functions

   .. autosummary::
      :toctree: metrics
      :template: function.rst

      mae
      mse
      mase