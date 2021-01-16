.. _lcss:

Longest Common Subsequence
==========================

Longest Common Subsequence (LCSS) [1]_ is a similarity measure between time series.
Let us consider two time series :math:`x = (x_0, \dots, x_{n-1})` and
:math:`y = (y_0, \dots, y_{m-1})` of respective lengths :math:`n` and
:math:`m`.
Here, all elements :math:`x_i` and :math:`y_j` are assumed to lie in the same
:math:`d`-dimensional space.
In ``tslearn``, such time series would be represented as arrays of respective
shapes `(n, d)` and `(m, d)` and LCSS can be computed using the following code:

.. code-block:: python

    from tslearn.metrics import lcss, lcss_path

    lcss_score = lcss(x, y)
    # Or, if the path is also an important information:
    path, lcss_score = lcss_path(x, y)


This is the algorithm at stake when invoking
:class:`tslearn.clustering.TimeSeriesKMeans` with
``metric="lcss"``.

Problem
--------------------

The similarity S between :math:`x` and :math:`y`, given an integer :math `\epsilon` and
a real number :math `\delta`, is formulated as follows:

.. math::

    S(x, y, \epsilon, \delta) = \frac{LCSS_{(\epsilon, \delta) (x, y)}}{\min(n, m)}


The constant :math:`\delta` controls how far in time we can go in order to match a given
point from one time-series to a point in another time-series. The constant :math:`\epsilon`
is the matching threshold.

Here, a path can be seen as the parts of the time series where the Euclidean
distance between them does not exceed a given threshold, i.e., they are close/similar.

Algorithmic solution
--------------------

There exists an :math:`O(n^2)` algorithm to compute the solution for this
problem (pseudo-code is provided for time series indexed from 1 for
simplicity):

.. code-block:: python

    def lcss(x, y):
       # Initialization
       for i = 0..n
           C[i, 0] = 0
       for j = 0..m
           C[0, j] = 0

       # Main loop
       for i = 1..n
            for j = 1..m
                if dist(x_i, x_j) <= epsilon and abs(j - i) <= delta:
                    C[i, j] = C[i-1, j-1] + 1
                else:
                    C[i, j] = max(C[i, j-1], C[i-1, j])

       return C[n, m]


Using a different ground metric
-------------------------------

By default, ``tslearn`` uses squared Euclidean distance as the base metric
(i.e. :math:`dist()` in the problem above is the
Euclidean distance). If one wants to use another ground metric, the code
would then be:

.. code-block:: python

    from tslearn.metrics import lcss_path_from_metric
    path, cost = lcss_path_from_metric(x, y, metric=compatible_metric)


Properties
----------

The Longest Common Subsequence holds the following properties:

* :math:`\forall x, y, LCSS(x, y) \geq 0`
* :math:`\forall x, y, LCSS(x, y) = LCSS(y, x)`
* :math:`\forall x, LCSS(x, x) = 0`

However, mathematically speaking, LCSS is not a valid measure since it does
not satisfy the triangular inequality.

Additional constraints
----------------------

One can set additional constraints to the set of acceptable paths.
These constraints typically consists in forcing paths to lie close to the
diagonal.

First, the Sakoe-Chiba band is parametrized by a radius :math:`r` (number of
off-diagonal elements to consider, also called warping window size sometimes), 
as illustrated below:

.. figure:: ../_static/img/sakoe_chiba.png
    :width: 30%
    :align: center

    :math:`n = m = 10, r = 3`. Diagonal is marked in grey for better
    readability.

The corresponding code would be:

.. code-block:: python

    from tslearn.metrics import lcss
    cost = lcss(x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=3)


Second, the Itakura parallelogram sets a maximum slope :math:`s` for alignment
paths, which leads to a parallelogram-shaped constraint:

.. figure:: ../_static/img/itakura.png
    :width: 30%
    :align: center

    :math:`n = m = 10, s = 2`. Diagonal is marked in grey for better
    readability.

The corresponding code would be:

.. code-block:: python

    from tslearn.metrics import lcss
    cost = lcss(x, y, global_constraint="itakura", itakura_max_slope=2.)


.. minigallery:: tslearn.metrics.lcss tslearn.metrics.lcss_path tslearn.metrics.lcss_path_from_metric
    :add-heading: Examples Involving LCSS variants
    :heading-level: -


.. raw:: html

    <div style="clear: both;" />

References
----------

.. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
       Similar Multidimensional Trajectories", In Proceedings of the
       18th International Conference on Data Engineering (ICDE '02).
       IEEE Computer Society, USA, 673.