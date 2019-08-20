from numpy.testing import assert_allclose

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'



def test_variable_length_knn():
    X = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3], [2, 5, 6, 7, 8, 9]])
    y = [0, 0, 1]

    clf = KNeighborsTimeSeriesClassifier(metric="dtw", n_neighbors=1)
    clf.fit(X, y)
    assert_allclose(clf.predict(X), [0, 0, 1])
