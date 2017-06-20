# -*- coding: utf-8 -*-
"""
Nearest neighbors
=================

This example illustrates the use of nearest neighbor methods for database search and classification tasks.

Script output::
    1. Nearest neighbour search
    Computed nearest neighbor indices (wrt DTW)
     [[10 12  2]
     [ 0 13  5]
     [ 0  1 13]
     [ 0 11  5]
     [16 18 12]
     [ 3 17  9]
     [12  2 16]
     [ 7  3 17]
     [12  2 10]
     [12  2 18]
     [12  8  2]
     [ 3 17  7]
     [18 19  2]
     [ 0 17 13]
     [ 9  3  7]
     [12  2  8]
     [ 3  7  9]
     [ 0  1 13]
     [18 10  2]
     [10 12  2]]
    First nearest neighbor class: [0 0 0 0 1 1 0 1 0 0 0 1 0 0 0 0 1 0 0 0]

    2. Nearest neighbour classification
    Correct classification rate: 1.0
"""

# Code source: Romain Tavenard
# License: BSD 3 clause

from __future__ import print_function
import numpy
from sklearn.metrics import accuracy_score

from tslearn.generators import random_walk_blobs
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries

numpy.random.seed(0)
n_ts_per_blob, sz, d, n_blobs = 20, 100, 1, 2
color_list=["r", "g"]

# Prepare data
X, y = random_walk_blobs(n_ts_per_blob=n_ts_per_blob, sz=sz, d=d, n_blobs=n_blobs)
scaler = TimeSeriesScalerMinMax(min=0., max=1.)  # Rescale time series
X_scaled = scaler.fit_transform(X)

indices_shuffle = numpy.random.permutation(n_ts_per_blob * n_blobs)
X_shuffle = X_scaled[indices_shuffle]
y_shuffle = y[indices_shuffle]

X_train = X_shuffle[:n_ts_per_blob * n_blobs // 2]
X_test = X_shuffle[n_ts_per_blob * n_blobs // 2:]
y_train = y_shuffle[:n_ts_per_blob * n_blobs // 2]
y_test = y_shuffle[n_ts_per_blob * n_blobs // 2:]

# Nearest neighbor search
knn = KNeighborsTimeSeries(n_neighbors=3, metric="dtw")
knn.fit(X_train, y_train)
dists, ind = knn.kneighbors(X_test)
print("1. Nearest neighbour search")
print("Computed nearest neighbor indices (wrt DTW)\n", ind)
print("First nearest neighbor class:", y_test[ind[:, 0]])

# Nearest neighbor classification
knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric="dtw")
knn_clf.fit(X_train, y_train)
predicted_labels = knn_clf.predict(X_test)
print("\n2. Nearest neighbour classification")
print("Correct classification rate:", accuracy_score(y_test, predicted_labels))
