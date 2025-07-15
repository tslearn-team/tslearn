# -*- coding: utf-8 -*-
"""
=====================================
Unsupervised Nearest Neighbors Search
=====================================

This example illustrates how to perform unsupervised k-nearest neighbors [1] search using 
:class:`~tslearn.neighbors.KNeighborsTimeSeries` to identify similar time series 
in the `GunPoint` dataset.

The distance between time series is calculated using Dynamic Time Warping (DTW) algorithm [2].

[1] `Wikipedia entry for the k-nearest neighbors algorithm
<https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_

[2] H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
for spoken word recognition". IEEE Transactions on Acoustics, Speech, and
Signal Processing, 26(1), 43-49 (1978).
"""
# sphinx_gallery_start_ignore
import warnings
warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

# Author: Romain Tavenard
# License: BSD 3 clause

##############################################################################
# Load the dataset
# ----------------
#
# In this example we use the `GunPoint dataset from the UCR/UEA archive
# <https://www.timeseriesclassification.com/description.php?Dataset=GunPoint>`_ .
#
# The dataset contains hand movement trajectories (x coordinates) 
# for two different actions performed by actors: drawing a gun from a hip holster and 
# pointing with a finger at a target. 
#
# The dataset is scaled to the [0, 1] range to ensure that time series
# are on a comparable scale before applying distance-based classification.
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMinMax

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("GunPoint")
X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
y_train = y_train - 1  # Convert to binary classes (0 and 1)
y_test = y_test - 1
labels = ["Gun", "Point"]

##############################################################################
# Nearest neighbor search 
# ------------------------
#
# Now we fit a k-nearest neighbors model to the training data using 
# :class:`~tslearn.neighbors.KNeighborsTimeSeries` and find the three nearest neighbors
# of test time series from different classes using Dynamic Time Warping (DTW) distance.
# Unlike supervised classification, this demonstrates how nearest neighbor search
# can identify similar patterns in an unsupervised manner.
from tslearn.neighbors import KNeighborsTimeSeries
knn = KNeighborsTimeSeries(n_neighbors=3, metric="dtw")
knn.fit(X_train, y_train)
dists, ind = knn.kneighbors(X_test)

##############################################################################
# We will plot the test sample and its three nearest neighbors for two test samples:
# one where all neighbors belong to class 0 (Gun) and another where all neighbors belong
# to class 1 (Point).
import numpy as np
import matplotlib.pyplot as plt

ind_0 = np.argmin(np.sum(y_train[ind], axis=1)) # Find test sample with class 0 neighbors only
ind_1 = np.argmax(np.sum(y_train[ind], axis=1)) # Find test sample with class 0 neighbors only

plt.figure(figsize=(10, 6))
for i, idx in enumerate([ind_0, ind_1]):
    plt.subplot(2, 1, i + 1)
    plt.plot(X_test[idx].ravel(), "k-", label="Test time series")
    for j in range(3):
        plt.plot(X_train[ind[idx, j]].ravel(), alpha=0.7, linestyle="dashed",
                 label=f"NN {j + 1} (class {labels[y_train[ind[idx, j]]]})")
    
    plt.ylabel("Hand position x (scaled)")
    plt.legend()

plt.suptitle("Nearest Neighbors for Test Time Series")
plt.xlabel("Time")
plt.tight_layout()
plt.show()

##############################################################################
# Conclusion
# ----------------
# The plots demonstrate the effectiveness of k-nearest neighbors search using DTW distance
# for time series pattern recognition.
#
# In the top subplot, we see the nearest neighbors have similar shape patterns which
# represent the "Gun" drawing movement. The test time series and its neighbors share
# a characteristic plateau pattern with fluctuations before and after.
# 
# In the bottom subplot, we see the nearest neighbors from the "Point" movement class.
# Notably, these neighbors are identified as nearest despite a significant shift in time,
# which highlights a key feature of DTW distance - its ability to handle temporal distortions.

