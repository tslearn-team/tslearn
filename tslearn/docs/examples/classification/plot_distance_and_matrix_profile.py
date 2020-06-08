# -*- coding: utf-8 -*-
"""
Distance and Matrix Profiles
============================
This example illustrates how the matrix profile is calculated. For each
segment of a timeseries with a specified length, the distances between
each subsequence and that segment are calculated. The smallest distance
that is not zero is then the corresponding value in the matrix profile. 
"""

# Author: Gilles Vandewiele
# License: BSD 3 clause

import numpy
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.matrix_profile import MatrixProfile

import warnings
warnings.filterwarnings('ignore')

# Set a seed to ensure determinism
numpy.random.seed(42)

# Load the Trace dataset
X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")

# Normalize the time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)

# Take the first time series
ts = X_train[0, :, :]

# We will take the spike as a segment
subseq_len = 20
start = 45
segment = ts[start:start + subseq_len]

# Create our matrix profile
matrix_profiler = MatrixProfile(subsequence_length=subseq_len, scale=True)
mp = matrix_profiler.fit_transform([ts]).flatten()

# Create a grid for our plots
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(7, 4)
fig_ax1 = fig.add_subplot(gs[:3, :])
fig_ax2 = fig.add_subplot(gs[3, 0])
fig_ax3 = fig.add_subplot(gs[3, 1:])
fig_ax4 = fig.add_subplot(gs[4:, :])

# Plot our timeseries
fig_ax1.plot(ts, c='b', label='time series')
fig_ax1.add_patch(patches.Rectangle((start, numpy.min(ts) - 0.1), subseq_len,
                                    numpy.max(ts) - numpy.min(ts) + 0.2,
                                    facecolor='b', alpha=0.25,
                                    label='segment'))
fig_ax1.axvline(start, c='b', linestyle='--', lw=2, alpha=0.5,
	            label='segment start')
fig_ax1.legend(loc='lower right', ncol=4, fontsize=8)
fig_ax1.set_title('The time series')

# Plot our segment
fig_ax2.plot(segment, c='b')
fig_ax2.set_title('Segment')

# Calculate a distance profile, which represents the distance from each
# subsequence of the time series and the segment
distances = []
scaler = TimeSeriesScalerMeanVariance()
for i in range(len(ts) - subseq_len):
    scaled_ts = scaler.fit_transform(ts[i:i+subseq_len].reshape(1, -1, 1))
    scaled_segment = scaler.fit_transform(segment.reshape(1, -1, 1))
    distances.append(numpy.linalg.norm(scaled_ts - scaled_segment))

second_min_ix = numpy.argsort(distances)[1]
second_min = distances[second_min_ix]

# Plot our distance profile
fig_ax3.plot(distances, c='b')
fig_ax3.set_title('Distance profile')
fig_ax3.scatter(second_min_ix, distances[second_min_ix],
                c='r', marker='x', s=50,
                label='2nd min dist = {}'.format(numpy.around(second_min, 3)))
fig_ax3.legend(loc='lower right', fontsize=8)

# Plot our matrix profile
fig_ax4.plot(mp, c='b')
fig_ax4.set_title('Matrix profile')
fig_ax4.scatter(start, mp[second_min_ix],
                c='r', marker='x', s=75,
                label='MP segment = {}'.format(numpy.around(mp[start], 3)))
fig_ax4.axvline(start, c='b', linestyle='--', lw=2, alpha=0.5,
	            label='segment start')
fig_ax4.legend(loc='lower right', fontsize=8)
plt.savefig('distance_matrix_profile.svg', format='svg')
plt.show()
