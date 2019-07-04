# -*- coding: utf-8 -*-
"""
Hyper-parameter tuning of a Pipeline with KNeighborsTimeSeriesClassifier
========================================================================

In this example, we demonstrate how it is possible to use the different
algorithms of tslearn in combination with sklearn utilities, such as
the `sklearn.pipeline.Pipeline` and `sklearn.model_selection.GridSearchCV`.

"""

# Author: Gilles Vandewiele
# License: BSD 3 clause

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.datasets import CachedDatasets

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import numpy as np

import matplotlib.pyplot as plt

# Our pipeline consists of two phases. First, data will be normalized using
# min-max normalization. Afterwards, it is fed to a KNN classifier. For the
# KNN classifier, we tune the n_neighbors and weights hyper-parameters.
n_splits = 2
pipeline = GridSearchCV(
    Pipeline([
            ('normalize', TimeSeriesScalerMinMax()),
            ('knn', KNeighborsTimeSeriesClassifier())
    ]),
    {'knn__n_neighbors': [5, 25], 'knn__weights': ['uniform', 'distance']},
    cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
    iid=True
)

X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")

# Keep only timeseries of class 0, 1 or 2
X_train = X_train[y_train < 4]
y_train = y_train[y_train < 4]

# Keep only the first 50 timeseries of both train and
# retain only a small amount of each of the timeseries
X_train, y_train = X_train[:50, 50:150], y_train[:50]

# Plot our timeseries
colors = ['k', 'g', 'b', 'r', 'c']
plt.figure()
for ts, label in zip(X_train, y_train):
    plt.plot(ts, c=colors[label], alpha=0.5)
plt.title('The timeseries in the training set')
plt.tight_layout()
plt.show()

# Fit our pipeline
pipeline.fit(X_train, y_train)
results = pipeline.cv_results_

# Print each possible configuration parameter and the out-of-fold accuracies
print('Fitted KNeighborsTimeSeriesClassifier on random walk blobs...')
for i in range(len(results['params'])):
    s = '{}\t'.format(results['params'][i])
    for k in range(n_splits):
        s += '{}\t'.format(results['split{}_test_score'.format(k)][i])
    print(s.strip())
