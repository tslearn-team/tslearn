# -*- coding: utf-8 -*-
"""
KShape
======

This example uses the KShape clustering method that is based on cross-correlation to cluster time series.
"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import KShape
from tslearn.generators import random_walk_blobs

numpy.random.seed(0)
X, y = random_walk_blobs(n_ts_per_blob=50, sz=128, d=1, n_blobs=3)

# Euclidean k-means
ks = KShape(n_clusters=3, n_init=5, verbose=False, random_state=0)
y_pred = ks.fit_predict(X)

own_colors = ["r", "g", "b"]

plt.figure()
for xx, yy in zip(X, y_pred):
    plt.plot(numpy.arange(128), xx, own_colors[yy] + "-")

plt.show()
