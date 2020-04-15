.. _mod-utils:

tslearn.utils
=============

.. automodule:: tslearn.utils

   
   
   .. rubric:: Generic functions

   .. autosummary::
      :toctree: utils
      :template: function.rst

      to_time_series
      to_time_series_dataset
      to_sklearn_dataset
      ts_size
      ts_zeros
      load_time_series_txt
      save_time_series_txt
      check_equal_size
      check_dims

   .. rubric:: Conversion functions

   The following functions are provided for the sake of
   interoperability between standard Python packages for time series.
   They allow conversion between `tslearn` format and other libraries' formats.

   .. autosummary::
      :toctree: utils_conv
      :template: function.rst

      to_pyts_dataset
      from_pyts_dataset
      to_sktime_dataset
      from_sktime_dataset
      to_cesium_dataset
      from_cesium_dataset
      to_seglearn_dataset
      from_seglearn_dataset
      to_tsfresh_dataset
      from_tsfresh_dataset
      to_stumpy_dataset
      from_stumpy_dataset
      to_pyflux_dataset
      from_pyflux_dataset








