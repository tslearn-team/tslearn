# -*- coding: utf-8 -*-
"""
Model Persistence
=================

Many tslearn models can be saved to disk and used for predictions
at a later time. This can be particularly useful when a model takes
a long time to train.

**Available formats:** hdf5, json, pickle

Save a model to disk::

    model.to_<format>

Load a model from disk::

    model.from_<format>

**Basic usage**

.. code-block:: python

    # Instantiate a model
    model = ModelClass(<hyper-parameters>)

    # Train the model
    model.fit(X_train)

    # Save the model to disk
    model.to_hdf5('./trained_model.hdf5')

    # Load model from disk
    model.from_hdf5('./trained_mode.hdf5')

    # Make predictions
    y = model.predict(X_test)

.. note::

    For the following models the training data are saved to disk and
    may result in a large model file if the trainig dataset is large:
    ``KNeighborsTimeSeries``, ``KNeighborsTimeSeriesClassifier``, and
    ``KernelKMeans``

"""

# Example using KShape
import os
import tempfile

import numpy
import matplotlib.pyplot as plt

from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

seed = 0
numpy.random.seed(seed)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

# Keep first 3 classes and 50 first time series
X_train = X_train[y_train < 4]
X_train = X_train[:50]
numpy.random.shuffle(X_train)
# For this method to operate properly, prior scaling is required
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
sz = X_train.shape[1]

# Instantiate k-Shape model
init=numpy.array([X_train[44], X_train[47], X_train[0]])
ks = KShape(n_clusters=3, verbose=True, init=init)

# Train
ks.fit(X_train)

with tempfile.TemporaryDirectory() as tmpdir:
    # Save model
    filename = os.path.join(tmpdir, "ks_trained.hdf5")
    if not os.path.isfile(filename):
        ks.to_hdf5(filename)

    # Load model
    trained_ks = KShape.from_hdf5(filename)

# Use loaded model to make predictions
y_pred = trained_ks.predict(X_train)

plt.figure()
for yi in range(3):
    plt.subplot(3, 1, 1 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi + 1))

plt.tight_layout()
plt.show()
