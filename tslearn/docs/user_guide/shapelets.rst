.. _shapelets:

Shapelets
=========

Shapelets are defined in [1]_ as "subsequences that are in some sense
maximally representative of a class".
Informally, if we assume a binary classification setting, a shapelet is
discriminant if it is **present** in most series of one class and absent from
series of the other class.
To assess the level of presence, one uses shapelet matches:

.. math::

    d(\mathbf{x}, \mathbf{s}) =
        \min_t \| \mathbf{x}_{t\rightarrow t+L} - \mathbf{s} \|_2

where :math:`L` is the length (number of timestamps) of shapelet
:math:`\mathbf{s}` and :math:`\mathbf{x}_{t\rightarrow t+L}` is the subsequence
extracted from time series :math:`\mathbf{x}` that starts at time index
:math:`t` and stops at :math:`t+L`.
If the above-defined distance is small enough, then
shapelet :math:`\textbf{s}` is supposed to be present in time series
:math:`\mathbf{x}`.

.. figure:: ../../_images/sphx_glr_plot_shapelet_locations_001.svg
    :width: 80%
    :align: center

    The distance from a time series to a shapelet is done by sliding the
    shorter shapelet over the longer time series and calculating the
    point-wise distances. The minimal distance found is returned.

In a classification setting, the goal is then to find the most discriminant
shapelets given some labeled time series data.
Shapelets can be mined from the training set [1]_ or learned using
gradient-descent.

Learning Time-series Shapelets
------------------------------

``tslearn`` provides an implementation of "Learning Time-series Shapelets",
introduced in [2]_, that is an instance of the latter category.
In Learning Shapelets,
shapelets are learned such
that time series represented in their shapelet-transform space (`i.e.` their
distances to each of the shapelets) are linearly separable.
A shapelet-transform representation of a time series :math:`\mathbf{x}` given
a set of shapelets :math:`\{\mathbf{s}_i\}_{i \leq k}` is the feature vector:
:math:`[d(\mathbf{x}, \mathbf{s}_1), \cdots, d(\mathbf{x}, \mathbf{s}_k)]`.
This is illustrated below with a two-dimensional example.


.. figure:: ../../_images/sphx_glr_plot_shapelet_distances_001.svg
    :width: 80%
    :align: center

    An example of how time series are transformed into linearly separable
    distances.


In ``tslearn``, in order to learn shapelets and transform timeseries to
their corresponding shapelet-transform space, the following code can be used:

.. code-block:: python

    from tslearn.shapelets import LearningShapelets

    model = LearningShapelets(n_shapelets_per_size={3: 2})
    model.fit(X_train, y_train)
    train_distances = model.transform(X_train)
    test_distances = model.transform(X_test)
    shapelets = model.shapelets_as_time_series_


A :class:`tslearn.shapelets.LearningShapelets` model has several
hyper-parameters, such as the maximum number of iterations and the batch size.
One important hyper-parameters is the ``n_shapelets_per_size``
which is a dictionary where the keys correspond to the desired lengths of the 
shapelets and the values to the desired number of shapelets per length. When 
set to ``None``, this dictionary will be determined by a 
:ref:`heuristic <fun-tslearn.shapelets.grabocka_params_to_shapelet_size_dict>`. 
After creating the model, we can ``fit`` the optimal shapelets 
using our training data. After a fitting phase, the distances can be calculated 
using the ``transform`` function. Moreover, you can easily access the 
learned shapelets by using the ``shapelets_as_time_series_`` attribute.

It is important to note that due to the fact that a technique based on
gradient-descent is used to learn the shapelets, our model can be prone
to numerical issues (e.g. exploding and vanishing gradients). For that
reason, it is important to normalize your data. This can be done before
passing the data to the
``fit``
and
``transform``
methods, by using our
:mod:`tslearn.preprocessing`
module but this can be done internally by the algorithm itself by setting the
``scale``
parameter.


.. minigallery:: tslearn.shapelets.LearningShapelets
    :add-heading: Examples Involving Shapelet-based Estimators
    :heading-level: -


.. raw:: html

    <div style="clear: both;" />

References
----------

.. [1] L. Ye & E. Keogh. Time series shapelets: a new primitive for data
       mining. SIGKDD 2009.
.. [2] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
