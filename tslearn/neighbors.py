"""
The :mod:`tslearn.neighbors` module gathers nearest neighbor algorithms using time series metrics.
"""

import numpy
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neighbors.base import KNeighborsMixin, _get_weights
from scipy import stats
from scipy.spatial.distance import cdist as scipy_cdist
from sklearn.utils.extmath import weighted_mode

from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset


class KNeighborsTimeSeriesMixin(KNeighborsMixin):
    """Mixin for k-neighbors searches on Time Series."""

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            The query time series.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int
            Number of neighbors to get (default is the value passed to the constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the distance to points, only present if return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        self_neighbors = False
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if X is None:
            X = self._fit_X
            self_neighbors = True
        else:
            X = to_time_series_dataset(X)
        if self.metric == "dtw":
            cdist_fun = cdist_dtw
        elif self.metric in ["euclidean", "sqeuclidean", "cityblock"]:
            cdist_fun = lambda X, Xp: scipy_cdist(X.reshape((X.shape[0], -1)),
                                                  Xp.reshape((Xp.shape[0], -1)),
                                                  metric=self.metric)
        else:
            raise ValueError("Unrecognized time series metric string: %s (should be one of 'dtw', 'euclidean', "
                             "'sqeuclidean' or 'cityblock')" % self.metric)
        full_dist_matrix = cdist_fun(X, self._fit_X)
        ind = numpy.argsort(full_dist_matrix, axis=1)

        if self_neighbors:
            ind = ind[:, 1:]
        if n_neighbors > full_dist_matrix.shape[1]:
            n_neighbors = full_dist_matrix.shape[1]
        ind = ind[:, :n_neighbors]

        n_ts = X.shape[0]
        sample_range = numpy.arange(n_ts)[:, None]
        dist = full_dist_matrix[sample_range, ind]

        if return_distance:
            return dist, ind
        else:
            return ind


class KNeighborsTimeSeries(KNeighborsTimeSeriesMixin, NearestNeighbors):
    """Unsupervised learner for implementing neighbor searches for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.
    metric : {'dtw', 'euclidean', 'sqeuclidean', 'cityblock'} (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [3, 3, 2, 0], [1, 2, 2, 4]]
    >>> knn = KNeighborsTimeSeries(n_neighbors=1).fit(time_series)
    >>> dist, ind = knn.kneighbors([[1, 1, 2, 2, 2, 3, 4]], return_distance=True)
    >>> dist
    array([[ 0.]])
    >>> ind
    array([[0]])
    >>> knn2 = KNeighborsTimeSeries(n_neighbors=10, metric="euclidean").fit(time_series)
    >>> ind = knn2.kneighbors(return_distance=False)
    >>> ind.shape
    (3, 2)
    """
    def __init__(self, n_neighbors=5, metric="dtw", metric_params=None):
        NearestNeighbors.__init__(self, n_neighbors=n_neighbors, algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        """
        self._fit_X = to_time_series_dataset(X)
        return self

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            The query time series.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        n_neighbors : int
            Number of neighbors to get (default is the value passed to the constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the distance to points, only present if return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        return KNeighborsTimeSeriesMixin.kneighbors(self,
                                                    X=X,
                                                    n_neighbors=n_neighbors,
                                                    return_distance=return_distance)


class KNeighborsTimeSeriesClassifier(KNeighborsClassifier, KNeighborsTimeSeriesMixin):
    """Classifier implementing the k-nearest neighbors vote for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.
    weights : str or callable, optional (default: 'uniform')
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point
          will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of distances, and returns an array of the same
          shape containing the weights.
    metric : one of the metrics allowed for class :class:`.KNeighborsTimeSeries` (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters.

    Examples
    --------
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric="dtw")
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]], y=[0, 0, 1])
    >>> clf.predict([1, 2.2, 3.5])
    array([0])
    """
    def __init__(self, n_neighbors=5, weights='uniform', metric="dtw", metric_params=None):
        KNeighborsClassifier.__init__(self, n_neighbors=n_neighbors, weights=weights, algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, )
            Target values.
        """
        self._fit_X = to_time_series_dataset(X)
        self._fit_y = numpy.array(y)

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.
        """
        X_ = to_time_series_dataset(X)
        neigh_dist, neigh_ind = self.kneighbors(X_)

        weights = _get_weights(neigh_dist, self.weights)

        if weights is None:
            mode, _ = stats.mode(self._fit_y[neigh_ind], axis=1)
        else:
            mode, _ = weighted_mode(self._fit_y[neigh_ind], weights, axis=1)

        return mode[:, 0]
