# -*- coding: utf-8 -*-
"""
k-means
=======

This example uses :math:`k`-means clustering for time series. Two variants of the algorithm are available: standard
Euclidean :math:`k`-means and DBA-:math:`k`-means (for DTW Barycenter Averaging).

"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.generators import random_walk_blobs

numpy.random.seed(0)
X, y = random_walk_blobs(n_ts_per_blob=50, sz=128, d=1, n_blobs=3)

# Euclidean k-means
km = TimeSeriesKMeans(n_clusters=3, n_init=5, verbose=False, random_state=0)
y_pred = km.fit_predict(X)

own_colors = ["r", "g", "b"]

plt.figure()
plt.subplot(3, 1, 1)
for xx, yy in zip(X, y_pred):
    plt.plot(numpy.arange(128), xx, own_colors[yy] + "-")
plt.title("Euclidean $k$-means")

# DBA-k-means
dba_km = TimeSeriesKMeans(n_clusters=3, n_init=5, metric="dtw", verbose=False)
y_pred = dba_km.fit_predict(X)

plt.subplot(3, 1, 2)
for xx, yy in zip(X, y_pred):
    plt.plot(xx, own_colors[yy] + "-")
plt.title("DBA $k$-means")

# DBA-k-means
sdtw_km = TimeSeriesKMeans(n_clusters=3, n_init=5, metric="softdtw", metric_params={"gamma_sdtw": 2.}, verbose=False)
y_pred = sdtw_km.fit_predict(X)

plt.subplot(3, 1, 3)
for xx, yy in zip(X, y_pred):
    plt.plot(xx, own_colors[yy] + "-")
plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()
