# -*- coding: utf-8 -*-
"""
LB_Keogh
========

This example illustrates the principle of time series enveloppe as used in LB_Keogh and estimates similarity between
time series using LB_Keogh.
"""

# Code source: Romain Tavenard
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

enveloppe_down, enveloppe_up = metrics.lb_enveloppe(dataset_scaled[0], radius=3)
plt.plot(numpy.arange(sz), dataset_scaled[0, :, 0], "r-")
plt.plot(numpy.arange(sz), enveloppe_down[:, 0], "g-")
plt.plot(numpy.arange(sz), enveloppe_up[:, 0], "g-")
plt.plot(numpy.arange(sz), dataset_scaled[1, :, 0], "k-")

plt.show()

print("LB_Keogh similarity: ", metrics.lb_keogh(dataset_scaled[1],
                                                enveloppe_candidate=(enveloppe_down, enveloppe_up)))
