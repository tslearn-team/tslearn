# -*- coding: utf-8 -*-
"""
Learning Shapelets
==================

This example illustrates the use of the "Learning Shapelets" method for a time
series classification task and ``tslearn``'s shapelet localization method.

More information on the method can be found at:
http://fs.ismll.de/publicspace/LearningShapelets/.
"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import ShapeletModel, \
    grabocka_params_to_shapelet_size_dict

# Set a seed to ensure determinism
numpy.random.seed(42)

# Load the Trace dataset
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

# Normalize each of the timeseries in the Trace dataset
X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)

# Get statistics of the dataset
n_ts, ts_sz = X_train.shape[:2]
n_classes = len(set(y_train))

# Set the number of shapelets per size as done in the original paper
shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts,
                                                       ts_sz=ts_sz,
                                                       n_classes=n_classes,
                                                       l=0.125,
                                                       r=1)

# Define the model and fit it using the training data
shp_clf = ShapeletModel(n_shapelets_per_size=shapelet_sizes,
                        weight_regularizer=.01,
                        max_iter=100,
                        verbose=0,
                        random_state=42)
shp_clf.fit(X_train, y_train)

# Get the number of extracted shapelets, the (minimal) distances from
# each of the timeseries to each of the shapelets, and the corresponding
# locations (index) where the minimal distance was found
n_shapelets = sum(shapelet_sizes.values())
distances = shp_clf.transform(X_train)
predicted_locations = shp_clf.locate(X_train)

plt.figure()
plt.title("Example locations of shapelet matches "
          "(%d shapelets extracted)".format(n_shapelets))

# Plot the test timeseries with the best matches with the shapelets
test_ts_id = numpy.argmin(numpy.sum(distances, axis=1))
plt.plot(X_train[test_ts_id].ravel())

# Plot the shapelets on their best-matching locations
for idx_shp, shp in enumerate(shp_clf.shapelets_):
    t0 = predicted_locations[test_ts_id, idx_shp]
    plt.plot(numpy.arange(t0, t0 + len(shp)), shp, linewidth=2)

plt.tight_layout()
plt.show()
