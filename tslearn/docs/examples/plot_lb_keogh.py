# -*- coding: utf-8 -*-
"""
LB_Keogh
========

This example illustrates the principle of time series envelope as used in LB_Keogh and estimates similarity between
time series using LB_Keogh.
"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn import metrics

numpy.random.seed(0)
n_ts, sz, d = 2, 100, 1
dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
dataset_scaled = scaler.fit_transform(dataset)

plt.figure()

envelope_down, envelope_up = metrics.lb_envelope(dataset_scaled[0], radius=3)
plt.plot(numpy.arange(sz), dataset_scaled[0, :, 0], "r-")
plt.plot(numpy.arange(sz), envelope_down[:, 0], "g-")
plt.plot(numpy.arange(sz), envelope_up[:, 0], "g-")
plt.plot(numpy.arange(sz), dataset_scaled[1, :, 0], "k-")

plt.show()

print("LB_Keogh similarity: ", metrics.lb_keogh(dataset_scaled[1],
                                                envelope_candidate=(envelope_down, envelope_up)))
