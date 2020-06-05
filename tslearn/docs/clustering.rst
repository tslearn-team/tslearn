.. _clustering:

Time Series Clustering
======================

Dedicated methods exist for time series clustering.
A first thing to understand is why there is a need for dedicated methods.

The following Figure illustrates this need (code to reproduce is available
:ref:`in the Gallery of Examples
<sphx_glr_auto_examples_clustering_plot_kmeans.py>`):

.. figure:: _static/img/kmeans.svg
    :width: 100%
    :align: center

    :math:`k`-means clustering with Euclidean distance. Each subfigure represents series from a given cluster and their centroid (in red).

This Figure is the result of a :math:`k`-means clustering that uses Euclidean
distance as a base metric.
One issue with this metric is that it is not invariant to time shifts, while
the dataset at stake clearly holds such invariants.

:math:`k`-means and Dynamic Time Warping
----------------------------------------

Hence, it would be profitable to use :math:`k`-means clustering with a metric
that has these invariants, such as :ref:`Dynamic Time Warping <dtw>`.

The :mod:`tslearn.clustering` module in ``tslearn`` offers that
option, which leads to better clusters and centroids:

.. figure:: _static/img/kmeans_dtw.svg
    :width: 100%
    :align: center

    :math:`k`-means clustering with Dynamic Time Warping. Each subfigure represents series from a given cluster and their centroid (in red).

First, clusters gather time series of similar shapes, which is due to the
ability of Dynamic Time Warping (DTW) to deal with time shifts, as explained
above.
Second, cluster centers (aka centroids) are computed with respect to DTW, hence
they allow to retrieve a sensible average shape whatever the temporal shifts
in the cluster (see :ref:`our dedicated User Guide section <dtw-barycenters>`
for more details on how these barycenters are computed).

In ``tslearn``, clustering a time series dataset with :math:`k`-means and a
dedicated time series metric is as easy as


.. code-block:: python

    from tslearn.clustering import TimeSeriesKMeans

    model = TimeSeriesKMeans(n_clusters=3, metric="dtw",
                             max_iter=10, random_state=seed)
    model.fit(X_train)

where ``X_train`` is the considered unlabelled dataset of time series.
The ``metric`` parameter can also be set to ``"softdtw"`` as an alternative
time series metric (`cf.`
:ref:`our User Guide section on soft-DTW <dtw-softdtw>`).


Kernel :math:`k`-means and Time Series Kernels
----------------------------------------------

Another option to deal with such time shifts is to rely on the kernel trick.
Indeed, [1]_ introduces a positive semidefinite kernel for time series,
inspired from DTW.
Then, the kernel :math:`k`-means algorithm [2]_, that is equivalent to a
:math:`k`-means
that would operate in the Reproducing Kernel Hilbert Space associated to the
chosen kernel, can be used:

.. figure:: _static/img/kernel_kmeans.svg
    :width: 100%
    :align: center

    Kernel :math:`k`-means clustering with Global Alignment Kernel. Each subfigure represents series from a given cluster.

One significant difference however is that cluster centers are never computed
explicitly, hence time series assignments to cluster are the only kind of
information available once the clustering is performed.

Note also that Global Alignment Kernel is closely related to soft-DTW [3]_.
As such, though it is related to alignment metrics such as Dynamic Time Warping, it is not invariant to time shifts [4]_.

K-Shape
-------

**TODO**


.. minigallery:: tslearn.clustering.TimeSeriesKMeans tslearn.clustering.GlobalAlignmentKernelKMeans tslearn.clustering.KShape
    :add-heading: Examples Using Clustering Estimators
    :heading-level: -


.. raw:: html

    <div style="clear: both;" />

References
----------

.. [1] M. Cuturi. "Fast Global Alignment Kernels," ICML 2011.

.. [2] I. S. Dhillon, Y. Guan & B. Kulis.
       "Kernel k-means, Spectral Clustering and Normalized Cuts," KDD 2004.

.. [3] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.

.. [4] H. Janati, M. Cuturi, A. Gramfort. "Spatio-Temporal Alignments: Optimal
       transport through space and time," AISTATS 2020
