"""
The :mod:`tslearn.barycenters` module gathers algorithms for time series
barycenter computation.

A barycenter (or *Fréchet mean*) is a time series :math:`b` which minimizes
the sum of squared distances to the time series of a given data set :math:`x`:

.. math:: \\min \\sum_i d( b, x_i )^2

Only the methods :func:`dtw_barycenter_averaging` and
:func:`softdtw_barycenter` can operate on variable-length time-series
(see :ref:`here<variable-length-barycenter>`).

See the :ref:`barycenter examples<sphx_glr_auto_examples_clustering_plot_barycenters.py>`
for an overview.
"""

# Code for soft DTW is by Mathieu Blondel under Simplified BSD license

from .utils import _set_weights
from .euclidean import euclidean_barycenter
from .dba import dtw_barycenter_averaging, \
    dtw_barycenter_averaging_subgradient, dtw_barycenter_averaging_petitjean
from .softdtw import softdtw_barycenter


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

__all__ = [
        "euclidean_barycenter",

        "dtw_barycenter_averaging", "dtw_barycenter_averaging_subgradient",
        "dtw_barycenter_averaging_petitjean",

        "softdtw_barycenter"
]
