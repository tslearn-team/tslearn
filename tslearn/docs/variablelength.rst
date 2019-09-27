Methods for variable-length time series datasets
================================================

This page lists machine learning methods in `tslearn` that are able to deal
with datasets containing time series of different lengths.
We also provide example usage for these methods using the following
variable-length time series dataset:

.. code-block:: python

    from tslearn.utils import to_time_series_dataset
    X = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3], [2, 5, 6, 7, 8, 9]])
    y = [0, 0, 1]

Classification
--------------

* :ref:`KNeighborsTimeSeriesClassifier <class-tslearn.neighbors.KNeighborsTimeSeriesClassifier>`
* :ref:`TimeSeriesSVC <class-tslearn.svm.TimeSeriesSVC>`

Examples
~~~~~~~~

.. code-block:: python

    from tslearn.neighbors import KNeighborsTimeSeriesClassifier
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=2)
    knn.fit(X, y)

.. code-block:: python

    from tslearn.svm import TimeSeriesSVC
    clf = TimeSeriesSVC(C=1.0, kernel="gak")
    clf.fit(X, y)

Regression
----------

* :ref:`TimeSeriesSVR <class-tslearn.svm.TimeSeriesSVR>`

Examples
~~~~~~~~

.. code-block:: python

    from tslearn.svm import TimeSeriesSVR
    clf = TimeSeriesSVR(C=1.0, kernel="gak")
    y_reg = [1.3, 5.2, -12.2]
    clf.fit(X, y_reg)

Clustering
----------

* :ref:`GlobalAlignmentKernelKMeans <class-tslearn.clustering.GlobalAlignmentKernelKMeans>`
* :ref:`TimeSeriesKMeans <class-tslearn.clustering.TimeSeriesKMeans>`
* :ref:`silhouette_score <fun-tslearn.clustering.silhouette_score>`

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

.. code-block:: python

    from tslearn.clustering import TimeSeriesKMeans, silhouette_score
    km = TimeSeriesKMeans(n_clusters=2, metric="dtw")
    labels = km.fit_predict(X)
    silhouette_score(X, labels, metric="dtw")

Barycenter computation
----------------------

* :ref:`dtw_barycenter_averaging <fun-tslearn.barycenters.dtw_barycenter_averaging>`
* :ref:`softdtw_barycenter <fun-tslearn.barycenters.softdtw_barycenter>`

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

Model selection
---------------

Also, model selection tools offered by `sklearn` can be used on variable-length
data, in a standard way, such as:

.. code-block:: python

    from sklearn.model_selection import KFold, GridSearchCV
    from tslearn.neighbors import KNeighborsTimeSeriesClassifier

    knn = KNeighborsTimeSeriesClassifier(metric="dtw")
    p_grid = {"n_neighbors": [1, 5]}

    cv = KFold(n_splits=2, shuffle=True, random_state=0)
    clf = GridSearchCV(estimator=knn, param_grid=p_grid, cv=cv)
    clf.fit(X, y)


Resampling
----------

* :ref:`TimeSeriesResampler <class-tslearn.preprocessing.TimeSeriesResampler>`

Finally, if you want to use a method that cannot run on variable-length time
series, one option would be to first resample your data so that all your
time series have the same length and then run your method on this resampled 
version of your dataset.

Note however that resampling will introduce temporal distortions in your 
data. Use with great care!

.. code-block:: python

    from tslearn.preprocessing import TimeSeriesResampler

    resampled_X = TimeSeriesResampler(sz=X.shape[1]).fit_transform(X)


