# -*- coding: utf-8 -*-
"""
1-NN with SAX + MINDIST
=======================

This example presents a comparison performs kNN with k=1 on SAX
transformations of the SyntheticControl dataset. MINDIST from the original
paper is used as a distance metric.
"""

# Author: Gilles Vandewiele
# License: BSD 3 clause

import numpy
import matplotlib.pyplot as plt
import time

from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.piecewise import PiecewiseAggregateApproximation, \
    SymbolicAggregateApproximation, \
    OneD_SymbolicAggregateApproximation

from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances, accuracy_score, \
    confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import norm

import warnings
warnings.filterwarnings('ignore')

numpy.random.seed(0)
# Generate a random walk time series
data_loader = UCR_UEA_datasets()
datasets = [
    ('SyntheticControl', 16),
    ('GunPoint', 64),
    ('OSULeaf', 128),
    ('Trace', 128),
    ('FaceFour', 128),
    ('Lightning2', 256),
    ('Lightning7', 128),
    ('ECG200', 32),
    ('Fish', 128),
    ('Plane', 64),
    ('Car', 256),
    ('Beef', 128),
    ('Coffee', 128),
    ('OliveOil', 256)
]

def print_table(accuracies, times):
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
        s += '{:>12}|'.format(eucl_error)
        s += '{:>12}|'.format(eucl_error)
        s += '{:>12}|'.format(time_euclidean)
        print(s.strip())

    print('-'*(len(columns) * 13 + 22))

accuracies = {}
times = {}
for dataset, w in datasets:
    X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)
    metric_params = {'n_segments': w, 'alphabet_size_avg': 10}
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric='sax',
                                         metric_params=metric_params)
    start = time.time()
    knn.fit(X_train, y_train)
    acc_sax = accuracy_score(y_test, knn.predict(X_test))
    time_sax = time.time() - start

    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric='euclidean')
    start = time.time()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc_euclidean = accuracy_score(y_test, predictions)
    time_euclidean = time.time() - start

    accuracies[dataset] = (acc_sax, acc_euclidean)
    times[dataset] = (time_sax, time_euclidean)

print_table(accuracies, times)
