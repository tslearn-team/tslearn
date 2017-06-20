# -*- coding: utf-8 -*-
"""
Barycenters
===========

Computing barycenters is a key operation for several ML techniques (e.g. clustering, ...).
This example shows two methods to compute barycenters of time series.
"""

# Code source: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.barycenters import EuclideanBarycenter, DTWBarycenterAveraging
from tslearn.generators import random_walk_blobs

n_ts, sz, d = 20, 128, 1

numpy.random.seed(0)
X, y = random_walk_blobs(n_ts_per_blob=n_ts, sz=sz, d=d, noise_level=2., n_blobs=1)

plt.figure()
plt.subplot(2, 1, 1)
for ts in X:
    plt.plot(numpy.arange(sz), ts, "k-", alpha=.2)
plt.plot(numpy.arange(sz), EuclideanBarycenter().fit(X), "r-", linewidth=2)
plt.title("Euclidean barycenter")

plt.subplot(2, 1, 2)
dba = DTWBarycenterAveraging(n_iter=100, verbose=False)
dba_bar = dba.fit(X)
for ts in X:
    plt.plot(numpy.arange(sz), ts, "k-", alpha=.2)
plt.plot(numpy.arange(sz), dba_bar, "r-", linewidth=2)
plt.title("DBA")

plt.show()
