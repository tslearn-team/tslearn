# -*- coding: utf-8 -*-
"""
Aligning discovered shapelets with timeseries
=============================================

This example illustrates the use of the "Learning Shapelets" method in order
to learn a collection of shapelets that linearly separates the timeseries.
In this example, we will extract two shapelets which are then used to
transform our input time series in a two-dimensional space. Moreover, we
plot the decision boundaries of our classifier for each of the different
classes.

More information on the method can be found at:
http://fs.ismll.de/publicspace/LearningShapelets/.
"""

# Author: Gilles Vandewiele
# License: BSD 3 clause

import numpy
from matplotlib import cm
import matplotlib.pyplot as plt

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.shapelets import LearningShapelets, \
    grabocka_params_to_shapelet_size_dict
from tensorflow.keras.optimizers import SGD

# Set a seed to ensure determinism
numpy.random.seed(42)

# Load the Trace dataset
X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")

# Sample some time series of each class
sampled_ix = []
for y in numpy.unique(y_train):
    class_ix = numpy.where(y_train == y)[0]
    sampled_ix.extend(list(numpy.random.choice(class_ix, replace=False,
                                               size=5)))
X_train = X_train[sampled_ix]
y_train = y_train[sampled_ix]

# Normalize the time series
X_train = TimeSeriesScalerMinMax().fit_transform(X_train)

# Get statistics of the dataset
n_ts, ts_sz = X_train.shape[:2]
n_classes = len(set(y_train))

# We will extract 1 shapelet and align it with a time series
shapelet_sizes = {20: 2}

# Define the model and fit it using the training data
shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                            weight_regularizer=0.0001,
                            optimizer=SGD(lr=0.9),
                            max_iter=1000,
                            verbose=0,
                            scale=False,
                            random_state=42)
shp_clf.fit(X_train, y_train)

# We will plot our distances in a 2D space
distances = shp_clf.transform(X_train).reshape((-1, 2))
weights, biases = shp_clf.model_.layers[-1].get_weights()

# Create a grid for our two shapelets on the left and distances on the right
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 3)
fig_ax1 = fig.add_subplot(gs[0, 0])
fig_ax2 = fig.add_subplot(gs[1, 0])
fig_ax3 = fig.add_subplot(gs[:, 1:])

# Plot our two shapelets on the left side
fig_ax1.plot(shp_clf.shapelets_[0])
fig_ax1.set_title('$s_1$')

fig_ax2.plot(shp_clf.shapelets_[1])
fig_ax2.set_title('$s_2$')

# Create a scatter plot of the 2D distances for the time series of each class.
viridis = cm.get_cmap('viridis', 4)
for i, y in enumerate(numpy.unique(y_train)):
    fig_ax3.scatter(distances[y_train == y][:, 0],
                    distances[y_train == y][:, 1],
                    c=[viridis(i / 3)] * numpy.sum(y_train == y),
                    label='Class {}'.format(y))

# Create a meshgrid of the decision boundaries
xmin = numpy.min(distances[:, 0]) - 0.1
xmax = numpy.max(distances[:, 0]) + 0.1
ymin = numpy.min(distances[:, 1]) - 0.1
ymax = numpy.max(distances[:, 1]) + 0.1
xx, yy = numpy.meshgrid(numpy.arange(xmin, xmax, (xmax - xmin)/200),
                        numpy.arange(ymin, ymax, (ymax - ymin)/200))
Z = []
for x, y in numpy.c_[xx.ravel(), yy.ravel()]:
    Z.append(numpy.argmax([biases[i] + weights[0][i]*x + weights[1][i]*y
                           for i in range(4)]))
Z = numpy.array(Z).reshape(xx.shape)
cs = fig_ax3.contourf(xx, yy, Z / 3, cmap=viridis, alpha=0.25)

fig_ax3.legend()
fig_ax3.set_xlabel('$d(x, s_1)$')
fig_ax3.set_ylabel('$d(x, s_2)$')
fig_ax3.set_xlim((xmin, xmax))
fig_ax3.set_ylim((ymin, ymax))
fig_ax3.set_title('Distance transformed time series')
plt.show()
