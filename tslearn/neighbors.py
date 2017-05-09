import numpy
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neighbors.base import KNeighborsMixin, _get_weights
from scipy import stats
from sklearn.utils.extmath import weighted_mode

from tslearn.metrics import dtw, cdist_dtw


class KNeighborsDynamicTimeWarpingMixin(KNeighborsMixin):
    """Mixin for k-neighbors searches using Dynamic Time Warping as the core metric."""

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
            Array representing the DTW to points, only present if return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        self_neighbors = False
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if X is None:
            X = self._fit_X
            self_neighbors = True
        full_dist_matrix = cdist_dtw(X, self._fit_X)
        ind = numpy.argsort(full_dist_matrix, axis=1)

        if self_neighbors:
            ind = ind[:, 1:]
        if n_neighbors > full_dist_matrix.shape[1]:
            n_neighbors = full_dist_matrix.shape[1]
        ind = ind[:, :n_neighbors]
        dist = full_dist_matrix[ind]

        if return_distance:
            return dist, ind
        else:
            return ind


class KNeighborsDynamicTimeWarpingClassifier(KNeighborsClassifier, KNeighborsDynamicTimeWarpingMixin):
    """Classifier implementing the k-nearest neighbors vote with Dynamic Time Warping as its core metric."""
    def __init__(self, n_neighbors=5, weights='uniform', **kwargs):
        KNeighborsClassifier.__init__(self, n_neighbors=n_neighbors, weights=weights, algorithm='brute', metric=dtw,
                                      **kwargs)

    def fit(self, X, y):
        self._fit_X = X
        self._fit_y = y

    def predict(self, X):
        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        if weights is None:
            mode, _ = stats.mode(self._fit_y[neigh_ind], axis=1)
        else:
            mode, _ = weighted_mode(self._fit_y[neigh_ind], weights, axis=1)

        return mode[:, 0]
