"""
The :mod:`tslearn.neighbors` module gathers nearest neighbor algorithms using
time series metrics.
"""

import numpy
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neighbors.base import KNeighborsMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist as scipy_cdist

from tslearn.metrics import cdist_dtw, cdist_soft_dtw, VARIABLE_LENGTH_METRICS
from tslearn.utils import (to_time_series_dataset, to_sklearn_dataset,
                           check_dims)

neighbors.VALID_METRICS['brute'].extend(['dtw', 'softdtw'])


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
            Number of neighbors to get (default is the value passed to the
            constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the distance to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        if self.metric_params is not None:
            metric_params = self.metric_params.copy()
            if "n_jobs" in metric_params.keys():
                del metric_params["n_jobs"]
        else:
            metric_params = {}
        self_neighbors = False
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if X is None:
            X = self._X_fit
            self_neighbors = True
        if self.metric == "precomputed":
            full_dist_matrix = X
        else:
            parallelize = False
            if self.metric == "dtw" or self.metric == cdist_dtw:
                cdist_fun = cdist_dtw
                parallelize = True
            elif self.metric == "softdtw" or self.metric == cdist_soft_dtw:
                cdist_fun = cdist_soft_dtw
            elif self.metric in ["euclidean", "sqeuclidean", "cityblock"]:
                def cdist_fun(X, Xp):
                    return scipy_cdist(X.reshape((X.shape[0], -1)),
                                       Xp.reshape((Xp.shape[0], -1)),
                                       metric=self.metric)
            else:
                raise ValueError("Unrecognized time series metric string: %s "
                                 "(should be one of 'dtw', 'softdtw', "
                                 "'euclidean', 'sqeuclidean' "
                                 "or 'cityblock')" % self.metric)

            if X.ndim == 2:  # sklearn-format case
                X = X.reshape((X.shape[0], -1, self._d))
                fit_X = self._X_fit.reshape((self._X_fit.shape[0],
                                             -1,
                                             self._d))
            else:
                fit_X = self._X_fit
            if parallelize:
                full_dist_matrix = cdist_fun(X, fit_X, n_jobs=self.n_jobs,
                                             **metric_params)
            else:
                full_dist_matrix = cdist_fun(X, fit_X, **metric_params)
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
    metric : {'dtw', 'softdtw', 'euclidean', 'sqeuclidean', 'cityblock'}
    (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure.
        DTW is described in more details in :mod:`tslearn.metrics`.
        Other metrics are described in `scipy.spatial.distance doc
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_.
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    Examples
    --------
    >>> time_series = to_time_series_dataset([[1, 2, 3, 4],
    ...                                       [3, 3, 2, 0],
    ...                                       [1, 2, 2, 4]])
    >>> knn = KNeighborsTimeSeries(n_neighbors=1).fit(time_series)
    >>> dataset = to_time_series_dataset([[1, 1, 2, 2, 2, 3, 4]])
    >>> dist, ind = knn.kneighbors(dataset, return_distance=True)
    >>> dist
    array([[0.]])
    >>> ind
    array([[0]])
    >>> knn2 = KNeighborsTimeSeries(n_neighbors=10,
    ...                             metric="euclidean").fit(time_series)
    >>> knn2.kneighbors(return_distance=False)
    array([[2, 1],
           [2, 0],
           [0, 1]])
    """
    def __init__(self, n_neighbors=5, metric="dtw", metric_params=None,
                 n_jobs=None):
        NearestNeighbors.__init__(self,
                                  n_neighbors=n_neighbors,
                                  algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        """
        X = check_array(X, allow_nd=True)
        self._X_fit = to_time_series_dataset(X)
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
            Number of neighbors to get (default is the value passed to the
            constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the distance to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        return KNeighborsTimeSeriesMixin.kneighbors(
            self,
            X=X,
            n_neighbors=n_neighbors,
            return_distance=return_distance)


class KNeighborsTimeSeriesClassifier(KNeighborsTimeSeriesMixin,
                                     KNeighborsClassifier):
    """Classifier implementing the k-nearest neighbors vote for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.
    weights : str or callable, optional (default: 'uniform')
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance. in this
          case, closer neighbors of a query point
          will have a greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an array of
          distances, and returns an array of the same
          shape containing the weights.
    metric : one of the metrics allowed for :class:`.KNeighborsTimeSeries`
    class (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure
    metric_params : dict or None (default: None)
        Dictionnary of metric parameters.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    Examples
    --------
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2, metric="dtw")
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
    ...                                      metric="dtw",
    ...                                      n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    """
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='dtw',
                 metric_params=None,
                 n_jobs=None):
        KNeighborsClassifier.__init__(self,
                                      n_neighbors=n_neighbors,
                                      weights=weights,
                                      algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, )
            Target values.
        """
        if self.metric in VARIABLE_LENGTH_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

        X = check_array(X,
                        allow_nd=True,
                        force_all_finite=(self.metric != "precomputed"))
        X = to_time_series_dataset(X)
        X = check_dims(X, X_fit=None)
        if self.metric == "precomputed" and hasattr(self, '_ts_metric'):
            self._ts_fit = X
            self._d = X.shape[2]
            self._X_fit = numpy.zeros((self._ts_fit.shape[0],
                                       self._ts_fit.shape[0]))
        else:
            self._X_fit, self._d = to_sklearn_dataset(X, return_dim=True)
        super(KNeighborsTimeSeriesClassifier, self).fit(self._X_fit, y)
        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric
        return self

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.
        """
        if self.metric in VARIABLE_LENGTH_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

            if self.metric_params is None:
                metric_params = {}
            else:
                metric_params = self.metric_params.copy()
                if "n_jobs" in metric_params.keys():
                    del metric_params["n_jobs"]
            check_is_fitted(self, '_ts_fit')
            X = check_array(X, allow_nd=True, force_all_finite=False)
            X = to_time_series_dataset(X)
            if self._ts_metric == "dtw":
                X_ = cdist_dtw(X, self._ts_fit, n_jobs=self.n_jobs,
                               **metric_params)
            elif self._ts_metric == "softdtw":
                X_ = cdist_soft_dtw(X, self._ts_fit, **metric_params)
            else:
                raise ValueError("Invalid metric recorded: %s" %
                                 self._ts_metric)
            pred = super(KNeighborsTimeSeriesClassifier, self).predict(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, self._X_fit, extend=False)
            return super(KNeighborsTimeSeriesClassifier, self).predict(X_)

    def predict_proba(self, X):
        """Predict the class probabilities for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.
        """
        if self.metric in VARIABLE_LENGTH_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

            if self.metric_params is None:
                metric_params = {}
            else:
                metric_params = self.metric_params.copy()
                if "n_jobs" in metric_params.keys():
                    del metric_params["n_jobs"]
            check_is_fitted(self, '_ts_fit')
            X = check_array(X, allow_nd=True, force_all_finite=False)
            X = to_time_series_dataset(X)
            if self._ts_metric == "dtw":
                X_ = cdist_dtw(X, self._ts_fit, n_jobs=self.n_jobs,
                               **metric_params)
            elif self._ts_metric == "softdtw":
                X_ = cdist_soft_dtw(X, self._ts_fit, **metric_params)
            else:
                raise ValueError("Invalid metric recorded: %s" %
                                 self._ts_metric)
            pred = super(KNeighborsTimeSeriesClassifier,
                         self).predict_proba(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, self._X_fit, extend=False)
            return super(KNeighborsTimeSeriesClassifier,
                         self).predict_proba(X_)

    def _get_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}
