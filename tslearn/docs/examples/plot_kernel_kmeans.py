# -*- coding: utf-8 -*-
"""
Kernel k-means
==============

This example uses Global Alignment kernel at the core of a kernel :math:`k`-means algorithm to perform time series
clustering.

"""

# Code source: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.metrics import sigma_gak, cdist_gak
from tslearn.generators import random_walk_blobs

numpy.random.seed(0)
X, y = random_walk_blobs(n_ts_per_blob=50, sz=128, d=1, n_blobs=3)

sigma = sigma_gak(X)

plt.figure()
for i, sigma in enumerate([sigma / 100, sigma, 100 * sigma]):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cdist_gak(X, sigma=sigma))
    plt.title("Gram matrix\n($\sigma = %.2f$)" % sigma)

plt.subplot2grid((2, 3), (1, 0), colspan=3)
gak_km = GlobalAlignmentKernelKMeans(n_clusters=3, sigma=sigma, n_init=20, verbose=False, random_state=0)
y_pred = gak_km.fit_predict(X)
if gak_km.X_fit_ is not None:
    own_colors = ["r", "g", "b"]
    for xx, yy in zip(X, y_pred):
        plt.plot(numpy.arange(128), xx, own_colors[yy] + "-")
plt.show()
