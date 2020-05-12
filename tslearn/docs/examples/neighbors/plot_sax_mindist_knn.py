# -*- coding: utf-8 -*-
"""
1-NN with SAX + MINDIST
=======================

This example presents a comparison between k-Nearest Neighbor runs with k=1.
It compares the use of:
* MINDIST (see [1]) on SAX representations of the data.
* Euclidean distance on the raw values of the time series.

The comparison is based on test accuracy using several benchmark datasets.

[1] Lin, Jessica, et al. "Experiencing SAX: a novel symbolic
    representation of time series." Data Mining and knowledge
    discovery 15.2 (2007): 107-144.
"""

# Author: Gilles Vandewiele
# License: BSD 3 clause

import warnings
import time

import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm

from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

from sklearn.base import clone
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


warnings.filterwarnings('ignore')


def print_table(accuracies, times):
    """Utility function to pretty print the obtained accuracies"""
    header_str = '|'
    header_str += '{:^20}|'.format('dataset')
    columns = ['sax error', 'sax time', 'eucl error', 'eucl time']
    for col in columns:
        header_str += '{:^12}|'.format(col)
    print(header_str)
    print('-'*(len(columns) * 13 + 22))

    for dataset in accuracies:
        acc_sax, acc_euclidean = accuracies[dataset]
        time_sax, time_euclidean = times[dataset]
        sax_error = numpy.around(1 - acc_sax, 5)
        eucl_error = numpy.around(1 - acc_euclidean, 5)
        time_sax = numpy.around(time_sax, 5)
        time_euclidean = numpy.around(time_euclidean, 5)
        s = '|'
        s += '{:>20}|'.format(dataset)
        s += '{:>12}|'.format(sax_error)
        s += '{:>12}|'.format(time_sax)
        s += '{:>12}|'.format(eucl_error)
        s += '{:>12}|'.format(time_euclidean)
        print(s.strip())

    print('-'*(len(columns) * 13 + 22))


# Set seed
numpy.random.seed(0)

# Defining dataset and the number of segments
data_loader = UCR_UEA_datasets()
datasets = [
    ('SyntheticControl', 16),
    ('GunPoint', 64),
    ('FaceFour', 128),
    ('Lightning2', 256),
    ('Lightning7', 128),
    ('ECG200', 32),
    ('Plane', 64),
    ('Car', 256),
    ('Beef', 128),
    ('Coffee', 128),
    ('OliveOil', 256)
]

# We will compare the accuracies & execution times of 1-NN using:
# (i) MINDIST on SAX representations, and
# (ii) euclidean distance on raw values
knn_sax = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric='sax')
knn_eucl = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric='euclidean')

accuracies = {}
times = {}
for dataset, w in datasets:
    X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)

    ts_scaler = TimeSeriesScalerMeanVariance()
    X_train = ts_scaler.fit_transform(X_train)
    X_test = ts_scaler.fit_transform(X_test)

    # Fit 1-NN using SAX representation & MINDIST
    metric_params = {'n_segments': w, 'alphabet_size_avg': 10}
    knn_sax = clone(knn_sax).set_params(metric_params=metric_params)
    start = time.time()
    knn_sax.fit(X_train, y_train)
    acc_sax = accuracy_score(y_test, knn_sax.predict(X_test))
    time_sax = time.time() - start

    # Fit 1-NN using euclidean distance on raw values
    start = time.time()
    knn_eucl.fit(X_train, y_train)
    acc_euclidean = accuracy_score(y_test, knn_eucl.predict(X_test))
    time_euclidean = time.time() - start

    accuracies[dataset] = (acc_sax, acc_euclidean)
    times[dataset] = (time_sax, time_euclidean)

print_table(accuracies, times)
