# -*- coding: utf-8 -*-
"""
Barycenters
===========

This example shows three methods to compute barycenters of time series.
"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.barycenters import euclidean_barycenter, dtw_barycenter_averaging, softdtw_barycenter
from tslearn.datasets import CachedDatasets

numpy.random.seed(0)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X = X_train[y_train == 2]

plt.figure()
plt.subplot(3, 1, 1)
for ts in X:
    plt.plot(ts.ravel(), "k-", alpha=.2)
plt.plot(euclidean_barycenter(X).ravel(), "r-", linewidth=2)
plt.title("Euclidean barycenter")

plt.subplot(3, 1, 2)
dba_bar = dtw_barycenter_averaging(X, max_iter=100, verbose=False)
for ts in X:
    plt.plot(ts.ravel(), "k-", alpha=.2)
plt.plot(dba_bar.ravel(), "r-", linewidth=2)
plt.title("DBA")

plt.subplot(3, 1, 3)
sdtw_bar = softdtw_barycenter(X, gamma=1., max_iter=100)
for ts in X:
    plt.plot(ts.ravel(), "k-", alpha=.2)
plt.plot(sdtw_bar.ravel(), "r-", linewidth=2)
plt.title("Soft-DTW barycenter ($\gamma$=1.)")

plt.tight_layout()
plt.show()
