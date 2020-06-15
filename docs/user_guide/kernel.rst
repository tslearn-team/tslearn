.. _kernel:

Kernel Methods
==============

In the following, we will discuss the use of kernels to compare time series.
A kernel :math:`k(\cdot, \cdot)` is such that there exists an unknown map
:math:`\Phi` such that:

.. math::

    k(\mathbf{x}, \mathbf{y}) =
        \left\langle
            \Phi(\mathbf{x}), \Phi(\mathbf{y})
        \right\rangle_{\mathcal{H}}

`i.e.` :math:`k(\cdot, \cdot)` is the inner product between :math:`\mathbf{x}`
and :math:`\mathbf{y}` in some (unknown) embedding space :math:`\mathcal{H}`.
In practice, :math:`k(\mathbf{x}, \mathbf{y})` will be large when
:math:`\mathbf{x}` and :math:`\mathbf{y}` are similar and close to 0 when they
are very dissimilar.

A large number of kernel methods from the machine learning literature rely on
the so-called `kernel trick`, that consists in performing computations in the
embedding space :math:`\mathcal{H}` without ever actually performing any
embedding.
As an example, one can compute distance between :math:`\mathbf{x}`
and :math:`\mathbf{y}` in :math:`\mathcal{H}` `via`:

.. math::

    \| \Phi(\mathbf{x}) - \Phi(\mathbf{y})\|_\mathcal{H}^2
        &= \left\langle \Phi(\mathbf{x}) - \Phi(\mathbf{y}),
                        \Phi(\mathbf{x}) - \Phi(\mathbf{y})
           \right\rangle_{\mathcal{H}} \\
        &= \left\langle \Phi(\mathbf{x}), \Phi(\mathbf{x})
           \right\rangle_{\mathcal{H}}  +
           \left\langle \Phi(\mathbf{y}), \Phi(\mathbf{y})
           \right\rangle_{\mathcal{H}}  - 2
           \left\langle \Phi(\mathbf{x}), \Phi(\mathbf{y})
           \right\rangle_{\mathcal{H}} \\
        &= k(\mathbf{x}, \mathbf{x}) + k(\mathbf{y}, \mathbf{y})
           - 2 k(\mathbf{x}, \mathbf{y})

Such computations are used, for example, in the kernel-:math:`k`-means
algorithm (see below).


Global Alignment Kernel
-----------------------

The Global Alignment Kernel (GAK) is a kernel that operates on time
series.

It is defined, for a given bandwidth :math:`\sigma`, as:

.. math::

    k(\mathbf{x}, \mathbf{y}) =
        \sum_{\pi \in \mathcal{A}(\mathbf{x}, \mathbf{y})}
            \prod_{i=1}^{ | \pi | }
                \exp \left( - \frac{ \left\| x_{\pi_1(i)} - y_{\pi_2{j}} \right\|^2}{2 \sigma^2} \right)

where :math:`\mathcal{A}(\mathbf{x}, \mathbf{y})` is the set of all possible
alignments between series :math:`\mathbf{x}` and :math:`\mathbf{y}`.

It is advised in [1]_ to set the bandwidth :math:`\sigma` as a multiple of a
simple estimate of the median distance of different points observed in
different time-series of your training set, scaled by the square root of the
median length of time-series in the set.
This estimate is made available in ``tslearn`` through
:ref:`fun-tslearn.metrics.sigma_gak`:

.. code-block:: python

    from tslearn.metrics import gak, sigma_gak

    sigma = sigma_gak(X)
    k_01 = gak(X[0], X[1], sigma=sigma)

Note however that, on long time series, this estimate can lead to numerical
overflows, which smaller values can avoid.

Finally, GAK is related to :ref:`softDTW <dtw-softdtw>` [3]_ through the
following formula:

.. math::

    k(\mathbf{x}, \mathbf{y}) =
        \exp \left(- \frac{\text{softDTW}_\gamma(\mathbf{x}, \mathbf{y})}{\gamma} \right)

where :math:`\gamma` is the hyper-parameter controlling softDTw smoothness,
which is related to the bandwidth parameter of GAK through
:math:`\gamma = 2 \sigma^2`.

.. _kernels-ml:

Clustering and Classification
-----------------------------

Kernel :math:`k`-means [2]_ is a method that uses the kernel trick to
implicitly perform :math:`k`-means clustering in the embedding space associated
to a kernel.
This method is discussed in
:ref:`our User Guide section dedicated to clustering <kernel-kmeans>`.

Kernels can also be used in classification settings.
:mod:`tslearn.svm` offers implementations of Support Vector Machines (SVM)
that accept GAK as a kernel.
This implementation heavily relies on ``scikit-learn`` and ``libsvm``.
One implication is that ``predict_proba`` and ``predict_log_proba`` methods
are computed based on cross-validation probability estimates, which has two
main implications, as discussed in more details in ``scikit-learn``'s
`user guide <https://scikit-learn.org/stable/modules/svm.html#scores-probabilities>`_:

1. setting the constructor option ``probability`` to ``True`` makes the ``fit``
step longer since it then relies on an expensive five-fold cross-validation;

2. the probability estimates obtained through ``predict_proba`` may be
inconsistent with the scores provided by ``decision_function`` and the
predicted class output by ``predict``.



.. minigallery:: tslearn.metrics.gak tslearn.metrics.cdist_gak tslearn.svm.TimeSeriesSVC tslearn.svm.TimeSeriesSVR tslearn.clustering.KernelKMeans
    :add-heading: Examples Using Kernel Methods
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
