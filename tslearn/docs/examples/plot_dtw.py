# -*- coding: utf-8 -*-
"""
DTW computation
===============

This example illustrates DTW computation between time series and plots the optimal alignment path.
"""

# Code source: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt
from matplotlib.pyplot import GridSpec

from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn import metrics

numpy.random.seed(0)
n_ts, sz, d = 2, 100, 1
dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
dataset_scaled = scaler.fit_transform(dataset)

path, sim = metrics.dtw_path(dataset_scaled[0], dataset_scaled[1])

matrix_path = numpy.zeros((sz, sz), dtype=numpy.int)
for i, j in path:
    matrix_path[i, j] = 1

plt.figure()

gs = GridSpec(1, 2, width_ratios=[2, 1], height_ratios=[1, 1])
ax0 = plt.subplot(gs[0])
ax0.plot(numpy.arange(sz), dataset_scaled[0, :, 0])
ax0.plot(numpy.arange(sz), dataset_scaled[1, :, 0])
ax1 = plt.subplot(gs[1])
ax1.imshow(matrix_path, cmap="gray_r")

plt.tight_layout()
plt.show()
