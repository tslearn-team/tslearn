.. _mod-metrics:

tslearn.metrics
===============

This modules delivers time-series specific metrics to be used at the core of
machine learning algorithms.
Dynamic Time Warping (DTW) is described in more details in `a dedicated page`_.

.. _a dedicated page: ../dtw.html

.. automodule:: tslearn.metrics



   .. rubric:: Functions

   .. autosummary::
      :toctree: metrics
      :template: function.rst

      cdist_dtw
      cdist_gak
      dtw
      dtw_path
      subsequence_path
      subsequence_cost_matrix
      dtw_subsequence_path
      gak
      soft_dtw
      cdist_soft_dtw
      cdist_soft_dtw_normalized
      lb_envelope
      lb_keogh
      sigma_gak
      gamma_soft_dtw
