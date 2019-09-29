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

numpy.random.seed(0)
# Generate a random walk time series
data_loader = UCR_UEA_datasets()
datasets = [
    ##('SyntheticControl', 16),
    ('GunPoint', 64),
    #('CBF', 32),
    #('FaceAll', 64),
    ##('OSULeaf', 128),
    #('SwedishLeaf', 32),
    #('FiftyWords', 128),
    ('Trace', 128),
    #('TwoPatterns', 32),
    ##('FaceFour', 128),
    ##('Lightning2', 256),
    ##('Lightning7', 128),
    ##('ECG200', 32),
    #('Adiac', 64),
    #('Yoga', 128),
    ##('Fish', 128),
    ##('Plane', 64),
    ##('Car', 256),
    ##('Beef', 128),
    ##('Coffee', 128),
    ##('OliveOil', 256)
]

def generate_lookup_table(a):
    return [norm.ppf(i * (1./a)) for i in range(1, a)]

def calc_distances(X, y=None):
    X = X.reshape((X.shape[0], X.shape[1]))
    cardinality = numpy.max(X) + 1
    n = X_train.shape[1]
    w = X.shape[1]
    table = generate_lookup_table(cardinality)

    def point_dist(i, j):
        i, j = int(i), int(j)
        if abs(i - j) <= 1:
            return 0
        else:
            return table[max(i, j) - 1] - table[min(i, j)]

    def sax_mindist(x, y):
        point_dists = [point_dist(x[i], y[i]) ** 2 for i in range(w)]
        return numpy.sqrt(n / w) * numpy.sqrt(numpy.sum(point_dists))


    return pairwise_distances(X, metric=sax_mindist)

def print_table(accuracies):
    header_str = '|'
    header_str += '{:^20}|'.format('dataset')
    columns = ['sax error', 'eucl error']
    for col in columns:
        header_str += '{:^12}|'.format(col)
    print(header_str)
    print('-'*(len(columns) * 13 + 22))

    for dataset in accuracies:
        acc_sax, acc_euclidean = accuracies[dataset]
        sax_error = numpy.around(1 - acc_sax, 5)
        eucl_error = numpy.around(1 - acc_euclidean, 5)
        s = '|'
        s += '{:>20}|'.format(dataset)
        s += '{:>12}|'.format(sax_error)
        s += '{:>12}|'.format(eucl_error)
        print(s.strip())

    print('-'*(len(columns) * 13 + 22))

for dataset, w in datasets:
    X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)
    metric_params = {'n_segments': w, 'alphabet_size_avg': 10}
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric='sax',
                                         metric_params=metric_params)
    knn.fit(X_train, y_train)
    knts_dist, knts_ind = knn.kneighbors(X_test)
    print(knn.predict(X_test))

    print(accuracy_score(y_test, knn.predict(X_test)))

    pipe = Pipeline([
        (
            'transform', 
            SymbolicAggregateApproximation(n_segments=w, alphabet_size_avg=10)
        ),
        (
            'distance', 
            FunctionTransformer(calc_distances, validate=False, pass_y=False)
        )
    ])

    all_X = numpy.vstack((X_train, X_test))
    distances = pipe.transform(all_X)

    # We only need the distances to the timeseries from the training set.
    # Both for the test and the training set.
    X_train_dist = distances[:len(X_train), :len(X_train)]
    X_test_dist = distances[len(X_train):, :len(X_train)]

    knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
    knn.fit(X_train_dist, y_train)
    print(knn.predict(X_test_dist))
    knn_dist, knn_ind = knn.kneighbors(X_test_dist)
    acc_sax = accuracy_score(y_test, knn.predict(X_test_dist))
    print(acc_sax)

    print(list(zip([x[0] for x in knts_dist], [x[0] for x in knn_dist])))
    print(list(zip([x[0] for x in knts_ind], [x[0] for x in knn_ind])))
    input()

    knn = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc_euclidean = accuracy_score(y_test, predictions)
    print(acc_euclidean)

    print()
    print()


# # Currently, I cannot append KNN to the pipeline, as it will raise problems
# # Due to the fact that it will transform X_test to a distance matrix
# # of (X_test.shape[0], X_test.shape[0])
# accuracies = {}
# for dataset, w in datasets:
#     X_train, y_train, X_test, y_test = data_loader.load_dataset(dataset)
#     pipe = Pipeline([
#         (
#             'transform', 
#             SymbolicAggregateApproximation(n_segments=w, alphabet_size_avg=10)
#         ),
#         (
#             'distance', 
#             FunctionTransformer(calc_distances, validate=False, pass_y=False)
#         )
#     ])

#     all_X = numpy.vstack((X_train, X_test))
#     distances = pipe.transform(all_X)

#     # We only need the distances to the timeseries from the training set.
#     # Both for the test and the training set.
#     X_train_dist = distances[:len(X_train), :len(X_train)]
#     X_test_dist = distances[len(X_train):, :len(X_train)]

#     knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
#     knn.fit(X_train_dist, y_train)
#     acc_sax = accuracy_score(y_test, knn.predict(X_test_dist))

#     knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
#     knn.fit(X_train.reshape(X_train.shape[:-1]), y_train)
#     predictions = knn.predict(X_test.reshape(X_test.shape[:-1]))
#     acc_euclidean = accuracy_score(y_test, predictions)

#     accuracies[dataset] = (acc_sax, acc_euclidean)

# print_table(accuracies)