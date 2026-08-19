# -*- coding: utf-8 -*-
"""
Silhouette analysis
===================

This example computes, for each time series of a dataset clustered with
:math:`k`-means, its individual silhouette coefficient [1]_. While the mean
silhouette coefficient gives a single scalar summarizing cluster quality, the
per-sample values reveal *which* time series are well-embedded in their
cluster and which ones are likely misassigned or outliers.

In the figure below, each bar corresponds to one time series, colored by its
cluster membership. Bars close to +1 indicate time series that are far from
neighboring clusters, bars near 0 indicate time series close to the decision
boundary between two clusters, and negative bars indicate time series that
may have been assigned to the wrong cluster.

.. [1] Peter J. Rousseeuw. "Silhouettes: a Graphical Aid to the
  Interpretation and Validation of Cluster Analysis". Computational and
  Applied Mathematics 20: 53-65, 1987.
"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans, silhouette_score, \
    silhouette_samples
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = X_train[y_train < 4]  # Keep first 3 classes
numpy.random.shuffle(X_train)
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
# Make time series shorter
X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)

# k-means clustering with DTW
km = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=seed)
y_pred = km.fit_predict(X_train)

# Per-sample and mean silhouette coefficients
samples = silhouette_samples(X_train, y_pred, metric="dtw")
score = silhouette_score(X_train, y_pred, metric="dtw")

print("Mean silhouette score: {:.4f}".format(score))

plt.figure()
y_lower = 10
for yi in range(3):
    cluster_samples = samples[y_pred == yi]
    cluster_samples.sort()
    y_upper = y_lower + cluster_samples.shape[0]
    plt.fill_betweenx(numpy.arange(y_lower, y_upper),
                      0., cluster_samples,
                      alpha=.7)
    plt.text(-0.03, y_lower + 0.5 * cluster_samples.shape[0],
             "%d" % (yi + 1))
    y_lower = y_upper + 5

plt.axvline(x=score, color="red", linestyle="--",
            label="Mean silhouette score")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")
plt.title("Silhouette plot for $k$-means (DTW) on the Trace dataset")
plt.legend(loc="best")
plt.yticks([])
plt.tight_layout()
plt.show()
