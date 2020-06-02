.. _shapelets:

Shapelets
=========

Shapelets are defined in [1]_ as "subsequences that are in some sense
maximally representative of a class".
Unformally, if we assume a binary classification setting, a shapelet is said
discriminant if it is **present** in most series of one class and absent from
series of the other class.
To assess the level of presence, one uses shapelet matches:

.. math::

    d(\mathbf{x}, \mathbf{s}) =
        \min_t \| \mathbf{x}_{t\rightarrow t+L} - \mathbf{s} \|_2

where :math:`L` is the length (number of timestamps) of shapelet
:math:`\mathbf{s}`. If the above-defined distance is small enough, then
shapelet :math:`\textbf{s}` is supposed to be present in time series
:math:`\mathbf{x}`.

**TODO: illustration here**

In a classification setting, the goal is then to find the most discriminant
shapelets given some time series data.
Shapelets can be sampled from the training set [1]_ or learned using
gradient-descent.

Learning Time-series Shapelets
------------------------------

``tslearn`` provides an implementation of "Learning Time-series Shapelets",
introduced in [2]_, that is an instance of the latter category.
In :ref:`Learning Shapelets <class-tslearn.shapelets.ShapeletModel>`,
shapelets are learned such
that time series represented in their shapelet-transform space are linearly
separable.
A shapelet-transform representation of a time series :math:`\mathbf{x}` given
a set of shapelets :math:`\{\mathbf{s}_i\}_{i \leq k}` is the feature vector:

.. math::

    [d(\mathbf{x}, \mathbf{s}_1), \cdots, d(\mathbf{x}, \mathbf{s}_k)]

In ``tslearn``, computing shapelet-transforms from a fitted model is

.. code-block:: python

    from tslearn.shapelets import ShapeletModel

    model = ShapeletModel(n_shapelets_per_size={3: 2})
    model.fit(X, y)
    shapelet_transform = model.transform(X)


**TODO: example of separability here in a 2d space**

Once fit, the model can shamelessly exhibit its shapelets:

.. code-block:: python

    from tslearn.shapelets import ShapeletModel

    model = ShapeletModel(n_shapelets_per_size={3: 2})
    model.fit(X, y)
    shapelets = model.shapelets_as_time_series_


.. include:: gen_modules/backreferences/tslearn.shapelets.ShapeletModel.examples


.. raw:: html

    <div style="clear: both;" />

References
----------

.. [1] L. Ye & E. Keogh. Time series shapelets: a new primitive for data
       mining. SIGKDD 2009.
.. [2] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
