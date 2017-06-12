import numpy
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neighbors.base import KNeighborsMixin, _get_weights
from scipy import stats
from sklearn.utils.extmath import weighted_mode

from tslearn.metrics import cdist_dtw, cdist_lr_dtw


class KNeighborsTimeSeries(KNeighborsMixin):
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
        if self.metric == "dtw":
            cdist_fun = cdist_dtw
        elif self.metric == "lr_dtw":
            cdist_fun = lambda x, y: cdist_lr_dtw(x, y, gamma=self.metric_params.get("gamma", 0.))
        else:
            raise ValueError("Unrecognized time series metric string: %s (should be one of 'dtw', 'lr_dtw')" %
                             self.metric)
        full_dist_matrix = cdist_fun(X, self._fit_X)
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


class KNeighborsTimeSeriesClassifier(KNeighborsClassifier, KNeighborsTimeSeries):
    """Classifier implementing the k-nearest neighbors vote for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.
    weights : str or callable, optional (default = 'uniform')
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this case, closer neighbors of a query point
          will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of distances, and returns an array of the same
          shape containing the weights.
    metric : {'dtw' or 'lr_dtw'} (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters. Recognized keys are `"gamma"` (which has default value 0.) for LR-DTW.
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
        self._fit_X = X
        self._fit_y = y

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.
        """
        neigh_dist, neigh_ind = self.kneighbors(X)

        weights = _get_weights(neigh_dist, self.weights)

        if weights is None:
            mode, _ = stats.mode(self._fit_y[neigh_ind], axis=1)
        else:
            mode, _ = weighted_mode(self._fit_y[neigh_ind], weights, axis=1)

        return mode[:, 0]
