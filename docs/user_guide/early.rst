.. _early:

Early Classification of Time Series
===================================

Early classification of time series is the task of performing a classification
as early as possible for an incoming time series, and decision about when
to trigger the decision is part of the prediction process.

Early Classification Cost Function
----------------------------------

Dachraoui et al. [1]_ introduces a composite loss function for early
classification of time series that balances earliness and accuracy.

The cost function is of the following form:

.. math::

    \mathcal{L}(\mathbf{x}_{\rightarrow t}, y, t, \boldsymbol{\theta}) =
        \mathcal{L}_c(\mathbf{x}_{\rightarrow t}, y, \boldsymbol{\theta})
        + \alpha t

where :math:`\mathcal{L}_c(\cdot,\cdot,\cdot)` is a
classification loss and :math:`t` is the time at which a
decision is triggered by the system (:math:`\mathbf{x}_{\rightarrow t}` is
time series :math:`\mathbf{x}` observed up to time :math:`t`).
In this setting, :math:`\alpha` drives the tradeoff between accuracy and
earliness and is supposed to be a hyper-parameter of the method.

The authors rely on (i) a clustering of the
training time series and (ii) individual classifiers :math:`m_t(\cdot)`
trained at all possible timestamps, so as to be able to predict,
at time :math:`t`, an expected cost for all future times :math:`t + \tau`
with :math:`\tau \geq 0`:

.. math::

    f_\tau(\mathbf{x}_{\rightarrow t}, y) =
        \sum_k \left[ P(C_k | \mathbf{x}_{\rightarrow t})
        \sum_i \left( P(y=i | C_k)
        \left( \sum_{j \neq i} P_{t+\tau}(\hat{y} = j | y=i, C_k)
        \right) \right)
        \right]
        + \alpha t

where:

* :math:`P(C_k | \mathbf{x}_{\rightarrow t})` is a soft-assignment weight of
  :math:`\mathbf{x}_{\rightarrow t}` to cluster :math:`C_k`;
* :math:`P(y=i | C_k)` is obtained from a contingency table that stores the
  number of training time series of each class in each cluster;
* :math:`P_{t+\tau}(\hat{y} = j | y=i, C_k)` is obtained through training time
  confusion matrices built on time series from cluster :math:`C_k` using
  classifier :math:`m_{t+\tau}(\cdot)`.

At test time, if a series is observed up to time :math:`t` and if, for all
positive :math:`\tau` we have
:math:`f_\tau(\mathbf{x}_{\rightarrow t}, y) \geq f_0(\mathbf{x}_{\rightarrow t}, y)`,
then a decision is made using classifier :math:`m_t(\cdot)`.

.. figure:: ../../_images/sphx_glr_plot_early_classification_002.svg
    :width: 80%
    :align: center

    Early classification. At test time, prediction is made at a timestamp such that the expected earliness-accuracy is optimized, which can hence vary between time series.

To use this early classifier in ``tslearn``, one can rely on the
:class:`tslearn.early_classification.NonMyopicEarlyClassifier` class:

.. code-block:: python

    from tslearn.early_classification import NonMyopicEarlyClassifier

    early_clf = NonMyopicEarlyClassifier(n_clusters=3,
                                         cost_time_parameter=1e-3,
                                         lamb=1e2,
                                         random_state=0)
    early_clf.fit(X_train, y_train)
    preds, times = early_clf.predict_class_and_earliness(X_test)

where ``cost_time_parameter`` is the :math:`\alpha` parameter presented above
and ``lamb`` is a trade-off parameter for the soft-assignment of partial series
to clusters :math:`P(C_k | \mathbf{x}_{\rightarrow t})` (when ``lamb`` tends to
infinity, the assignment tends to hard-assignment, and when ``lamb`` is set to
0, equal probabilities are obtained for all clusters).

.. minigallery:: tslearn.early_classification.NonMyopicEarlyClassifier
    :add-heading: Examples Involving Early Classification Estimators
    :heading-level: -


.. raw:: html

    <div style="clear: both;" />

References
----------

.. [1] A. Dachraoui, A. Bondu and A. Cornuejols.
       "Early classification of time series as a non myopic sequential decision
       making problem," ECML/PKDD 2015
