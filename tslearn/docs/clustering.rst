.. _clustering:

Time Series Clustering
======================

Dedicated methods exist for time series clustering.
A first thing to understand is why there is a need for dedicated methods.

The following Figure illustrates this need (code to reproduce is available in
the Gallery of Examples (**TODO link**):

**TODO img notebook**

This Figure is the result of a :math:`k`-means clustering that uses Euclidean
distance as a base metric.
One issue with this metric is that it is not invariant to time shifts, while
the dataset at stake clearly holds such invariants.

:math:`k`-means and Dynamic Time Warping
----------------------------------------

Hence, it would be profitable to use :math:`k`-means clustering with a metric
that has these invariants, such as :ref:`Dynamic Time Warping <dtw>`.

The :ref:`clustering <mod-clustering>` module in ``tslearn`` offers that
option, which leads to better clusters and centroids:

**TODO fig**

First, clusters gather time series of similar shapes, which is due to the
ability of Dynamic Time Warping (DTW) to deal with time shifts, as explained
above.
Second, cluster centers (aka centroids) are computed with respect to DTW, hence
they allow to retrieve an average shape whatever the temporal shifts in the
cluster (see Section DTW barycenters for more details on how these barycenters
are computed **TODO**).

In ``tslearn``, clustering a time series dataset with :math:`k`-means and a
dedicated time series metric is as easy as


.. code-block:: python

    from tslearn.clustering import TimeSeriesKMeans

    model = TimeSeriesKMeans(n_clusters=3, metric="dtw",
                             max_iter=10, random_state=seed)
    model.fit(X_train)

where ``X_train`` is the considered unlabelled dataset of time series.
The ``metric`` parameter can also be set to ``"softdtw"`` as an alternative
time series metric (`cf.` **TODO link to DTW section on softDTW**).


Kernel :math:`k`-means and Time Series Kernels
----------------------------------------------

Another option to deal with such time shifts is to rely on the kernel trick.
Indeed, [1]_ introduces a positive semidefinite kernel for time series,
inspired from DTW.
Then, the kernel :math:`k`-means algorithm [2]_, that is equivalent to a
:math:`k`-means
that would operate in the Reproducing Kernel Hilbert Space associated to the
chosen kernel, can be used:

**TODO fig**

One significant difference however is that cluster centers are never built
explicitly, hence time series assignments to cluster are the only kind of
information available once the clustering is performed.

K-Shape
-------

**TODO**



.. include:: gen_modules/backreferences/tslearn.clustering.examples


.. raw:: html

    <div style="clear: both;" />

References
----------

.. [1] M. Cuturi. Fast Global Alignment Kernels. ICML 2011.

.. [2] I. S. Dhillon, Y. Guan & B. Kulis.
       Kernel k-means, Spectral Clustering and Normalized Cuts. KDD 2004.
