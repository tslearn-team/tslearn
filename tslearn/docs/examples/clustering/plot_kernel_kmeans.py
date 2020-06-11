# -*- coding: utf-8 -*-
"""
Kernel k-means
==============

This example uses Global Alignment kernel (GAK, [1]) at the core of a kernel
:math:`k`-means algorithm [2] to perform time series clustering.

Note that, contrary to :math:`k`-means, a centroid cannot be computed when
using kernel :math:`k`-means. However, one can still report cluster
assignments, which is what is provided here: each subfigure represents the set
of time series from the training set that were assigned to the considered
cluster.

[1] M. Cuturi, "Fast global alignment kernels," ICML 2011.

[2] I. S. Dhillon, Y. Guan, B. Kulis. Kernel k-means, Spectral Clustering and \
Normalized Cuts. KDD 2004.
"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import KernelKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
# Keep first 3 classes
X_train = X_train[y_train < 4]
numpy.random.shuffle(X_train)
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
sz = X_train.shape[1]

gak_km = KernelKMeans(n_clusters=3,
                      kernel="gak",
                      kernel_params={"sigma": "auto"},
                      n_init=20,
                      verbose=True,
                      random_state=seed)
y_pred = gak_km.fit_predict(X_train)

plt.figure()
for yi in range(3):
    plt.subplot(3, 1, 1 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()
plt.show()
