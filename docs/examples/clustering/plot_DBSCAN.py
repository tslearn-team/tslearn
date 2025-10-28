# -*- coding: utf-8 -*-
"""
DBSCAN
======


"""
import numpy as np

from sklearn import metrics
from sklearn.utils import check_random_state

import matplotlib.pyplot as plt

from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.clustering import TimeSeriesDBSCAN,silhouette_score

rs = check_random_state(0)

nb_blobs = 3
X, y = random_walk_blobs(n_ts_per_blob=20, sz=64, n_blobs=nb_blobs, noise_level=0.5, random_state=rs)

# Pollute 10 ts
polluted_ts_indexes = rs.choice(X.shape[0], 10, replace=False)
for i in polluted_ts_indexes:
    # nb points impacted
    polluted_sample_indexes = rs.choice(X.shape[1], rs.randint(0, X.shape[1]), replace=False)
    X[i, polluted_sample_indexes] += rs.randint(-5, 5, size=polluted_sample_indexes.shape).reshape(-1, 1)

X = TimeSeriesScalerMinMax().fit_transform(X)

fig = plt.figure(1, figsize=(12, 12), layout="compressed")
axs = fig.subplots(len(np.unique(y)), 1, sharex=True)
for i, ax in enumerate(axs):
    for x in X[y==i]:
        ax.plot(x)
fig.suptitle("Polluted dataset")
plt.show()

##############################################################################
# Estimator fitting
# -----------------

db = TimeSeriesDBSCAN(eps=0.4, min_ts=11).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print(f"Homogeneity: {metrics.homogeneity_score(y, labels):.3f}")
print(f"Completeness: {metrics.completeness_score(y, labels):.3f}")
print(f"V-measure: {metrics.v_measure_score(y, labels):.3f}")
print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(y, labels):.3f}")
print(
    "Adjusted Mutual Information:"
    f" {metrics.adjusted_mutual_info_score(y, labels):.3f}"
)
print(f"Silhouette Coefficient: {silhouette_score(X, labels, metric=db.metric):.3f}")

##############################################################################
# Core and noise elements
# -----------------------

core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_ts_indices_] = True

fig = plt.figure(2, figsize=(12, 12),layout="compressed")
axs = fig.subplots(n_clusters_, 1, sharex=True)

for i, ax in enumerate(axs):
    class_member_mask = y == i

    # Core TS
    for x in X[class_member_mask & core_samples_mask]:
        ax.plot(x, "g")

    # Non cores TS, possibly considered as noise
    mask = class_member_mask & ~core_samples_mask
    for x, label in zip(X[mask], labels[mask]):
        ax.plot(x, "r" if label == -1 else ":k", alpha=1 if label == -1 else 0.5)
fig.suptitle("Core and noise elements")
plt.show()
