# -*- coding: utf-8 -*-
"""
SVM and GAK
===========

This example illustrates the use of the global alignment kernel for support vector classification.

TODO: cite the paper
"""

# Author: Romain Tavenard
# License: BSD 3 clause

import numpy

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.svm import TimeSeriesSVC

numpy.random.seed(0)
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
X_test = TimeSeriesScalerMinMax().fit_transform(X_test)

clf = TimeSeriesSVC(kernel="gak", gamma=.1, sz=X_train.shape[1], d=X_train.shape[2])
clf.fit(X_train, y_train)
print("Correct classification rate:", clf.score(X_test, y_test))
