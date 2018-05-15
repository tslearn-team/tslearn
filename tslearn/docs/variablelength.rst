Methods for variable-length time series datasets
================================================

This page lists machine learning methods in `tslearn` that are able to deal with datasets containing time series of different lengths.
We also provide example usage for these methods using the following variable-length time series dataset:

.. code-block:: python

    from tslearn.utils import to_time_series_dataset
    X = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3], [2, 5, 6, 7, 8, 9]])

Clustering
----------

* :ref:`GlobalAlignmentKernelKMeans <class-gakkmeans>`
* :ref:`TimeSeriesKMeans <class-dbakmeans>`

Examples
~~~~~~~~

.. code-block:: python

    from tslearn.clustering import GlobalAlignmentKernelKMeans
    gak_km = GlobalAlignmentKernelKMeans(n_clusters=2)
    labels_gak = gak_km.fit_predict(X)

.. code-block:: python

    from tslearn.clustering import TimeSeriesKMeans
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
    labels = km.fit_predict(X)
    km_bis = TimeSeriesKMeans(n_clusters=2, metric="softdtw")
    labels_bis = km_bis.fit_predict(X)

Supervised classification
-------------------------

* :ref:`KNeighborsTimeSeriesClassifier <knn-clf>`

Example
~~~~~~~

.. code-block:: python

    from tslearn.neighbors import KNeighborsTimeSeriesClassifier
    clf = KNeighborsTimeSeriesClassifier(metric="dtw")
    labels = clf.fit_predict(X)

Barycenter computation
----------------------

* :ref:`DTWBarycenterAveraging <class-dba>`
* :ref:`SoftDTWBarycenter <class-softdtw>`

Examples
~~~~~~~~

.. code-block:: python

    from tslearn.barycenters import dtw_barycenter_averaging
    bar = dtw_barycenter_averaging(X, barycenter_size=3)

.. code-block:: python

    from tslearn.barycenters import softdtw_barycenter
    from tslearn.utils import ts_zeros
    initial_barycenter = ts_zeros(sz=5)
    bar = softdtw_barycenter(X, init=initial_barycenter)
