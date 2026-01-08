# -*- coding: utf-8 -*-
"""
DBSCAN
======

This example illustrates density-based spatial clustering of applications with
noise (DBSCAN) for time series.
This method, based on finding high density cores, does not require specifying the number of
clusters and can identify outliers. The implementation relies on scikit-learn DBSCAN with
time series metrics support.

"""
import numpy as np

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesDBSCAN
from tslearn.datasets import CachedDatasets

X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")

# Keep only timeseries of class 1 and 2 plus 1 outlier from class 3
X_train = np.concatenate((
    X_train[y_train == 1][:10],
    X_train[y_train == 2][:10],
    X_train[y_train == 3][0].reshape(1, -1, 1),
))
y_train = np.concatenate((
    y_train[y_train == 1][:10],
    y_train[y_train == 2][:10],
    np.array([3]),
))


##############################################################################
# Estimator fitting
# -----------------

model = Pipeline([
    ('normalize', TimeSeriesScalerMinMax()),
    ('dbscan', TimeSeriesDBSCAN(eps=0.4, min_ts=10))
    ])
model = model.fit(X_train)
labels = model[-1].labels_
core_ts_indices = model[-1].core_ts_indices_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

##############################################################################
# Core and noise
# --------------

fig = plt.figure(figsize=(12, 4),layout="compressed")

core_samples_mask = np.full_like(labels, False, dtype=bool)
core_samples_mask[model[-1].core_ts_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_clusters_)]

for label in np.unique(labels):
    if label==-1:
        plt.plot(X_train[labels==label].squeeze(), ":k", alpha=1, label="Outlier")
    else:
        plt.plot(X_train[(labels == label) & core_samples_mask].squeeze().T, color=colors[label], alpha=0.5,
                 ls='-', label=f"Core TS of cluster {label}")
        plt.plot(X_train[(labels == label) & ~core_samples_mask].squeeze().T, color=colors[label], alpha=0.5,
                 ls='--', label=f"Non core TS of cluster {label}")

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='lower right')

fig.suptitle("Clustering results")
plt.show()
