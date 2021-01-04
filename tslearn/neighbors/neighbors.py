import numpy
from sklearn import neighbors
from sklearn.neighbors import (KNeighborsClassifier, NearestNeighbors,
                               KNeighborsRegressor)
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist as scipy_cdist

from tslearn.metrics import cdist_dtw, cdist_ctw, cdist_soft_dtw, \
    cdist_sax, TSLEARN_VALID_METRICS
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.utils import (to_time_series_dataset, to_sklearn_dataset,
                           check_dims)
from tslearn.bases import BaseModelPackage

neighbors.VALID_METRICS['brute'].extend(['dtw', 'softdtw', 'sax', 'ctw'])


class KNeighborsTimeSeriesMixin():
    """Mixin for k-neighbors searches on Time Series."""

    def _sax_preprocess(self, X, n_segments=10, alphabet_size_avg=4,
                        scale=False):
        # Now SAX-transform the time series
        if not hasattr(self, '_sax') or self._sax is None:
            self._sax = SymbolicAggregateApproximation(
                n_segments=n_segments,
                alphabet_size_avg=alphabet_size_avg,
                scale=scale
            )

        X = to_time_series_dataset(X)
        X_sax = self._sax.fit_transform(X)

        return X_sax

    def _get_metric_params(self):
        if self.metric_params is None:
            metric_params = {}
        else:
            metric_params = self.metric_params.copy()
        if "gamma_sdtw" in metric_params.keys():
            metric_params["gamma"] = metric_params["gamma_sdtw"]
            del metric_params["gamma_sdtw"]
        if "n_jobs" in metric_params.keys():
            del metric_params["n_jobs"]
        if "verbose" in metric_params.keys():
            del metric_params["verbose"]
        return metric_params

    def _precompute_cross_dist(self, X, other_X=None):
        if other_X is None:
            other_X = self._ts_fit

        self._ts_metric = self.metric
        self.metric = "precomputed"

        metric_params = self._get_metric_params()

        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = to_time_series_dataset(X)

        if self._ts_metric == "dtw":
            X_ = cdist_dtw(X, other_X, n_jobs=self.n_jobs,
                           **metric_params)
        elif self._ts_metric == "ctw":
            X_ = cdist_ctw(X, other_X, **metric_params)
        elif self._ts_metric == "softdtw":
            X_ = cdist_soft_dtw(X, other_X, **metric_params)
        elif self._ts_metric == "sax":
            X = self._sax_preprocess(X, **metric_params)
            X_ = cdist_sax(X, self._sax.breakpoints_avg_,
                           self._sax._X_fit_dims_[1], other_X,
                           n_jobs=self.n_jobs)
        else:
            raise ValueError("Invalid metric recorded: %s" %
                             self._ts_metric)

        return X_

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
        self_neighbors = False
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if X is None:
            X = self._X_fit
            self_neighbors = True
        if self.metric == "precomputed":
            full_dist_matrix = X
        else:

            if X.ndim == 2:  # sklearn-format case
                X = X.reshape((X.shape[0], -1, self._d))
                fit_X = self._X_fit.reshape((self._X_fit.shape[0],
                                             -1,
                                             self._d))
            elif hasattr(self, '_ts_fit') and self._ts_fit is not None:
                fit_X = self._ts_fit
            else:
                fit_X = self._X_fit

            if (self.metric in TSLEARN_VALID_METRICS or
                    self.metric in [cdist_dtw, cdist_ctw,
                                    cdist_soft_dtw, cdist_sax]):
                full_dist_matrix = self._precompute_cross_dist(X,
                                                               other_X=fit_X)
            elif self.metric in ["euclidean", "sqeuclidean", "cityblock"]:
                full_dist_matrix = scipy_cdist(X.reshape((X.shape[0], -1)),
                                               fit_X.reshape((fit_X.shape[0],
                                                              -1)),
                                               metric=self.metric)
            else:
                raise ValueError("Unrecognized time series metric string: %s "
                                 "(should be one of 'dtw', 'softdtw', "
                                 "'sax', 'euclidean', 'sqeuclidean' "
                                 "or 'cityblock')" % self.metric)

        # Code similar to sklearn (sklearn/neighbors/base.py), to make sure
        # that TimeSeriesKNeighbor~(metric='euclidean') has the same results as
        # feeding a distance matrix to sklearn.KNeighbors~(metric='euclidean')
        kbin = min(n_neighbors - 1, full_dist_matrix.shape[1] - 1)
        # argpartition will make sure the first `kbin` entries are the
        # `kbin` smallest ones (but in arbitrary order) --> complexity: O(n)
        ind = numpy.argpartition(full_dist_matrix, kbin, axis=1)

        if self_neighbors:
            ind = ind[:, 1:]
        if n_neighbors > full_dist_matrix.shape[1]:
            n_neighbors = full_dist_matrix.shape[1]
        ind = ind[:, :n_neighbors]

        n_ts = X.shape[0]
        sample_range = numpy.arange(n_ts)[:, None]
        # Sort the `kbin` nearest neighbors according to distance
        ind = ind[
            sample_range, numpy.argsort(full_dist_matrix[sample_range, ind])]
        dist = full_dist_matrix[sample_range, ind]

        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric

        if return_distance:
            return dist, ind
        else:
            return ind


class KNeighborsTimeSeries(KNeighborsTimeSeriesMixin, NearestNeighbors,
                           BaseModelPackage):
    """Unsupervised learner for implementing neighbor searches for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.

    metric : {'dtw', 'softdtw', 'ctw', 'euclidean', 'sqeuclidean', \
              'cityblock',  'sax'} (default: 'dtw')
        Metric to be used at the core of the nearest neighbor procedure.
        DTW and SAX are described in more detail in :mod:`tslearn.metrics`.
        When SAX is provided as a metric, the data is expected to be
        normalized such that each time series has zero mean and unit
        variance. Other metrics are described in `scipy.spatial.distance doc
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_.

    metric_params : dict or None (default: None)
        Dictionary of metric parameters.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` and `verbose` keys passed in `metric_params`
        are overridden by the `n_jobs` and `verbose` arguments.
        For 'sax' metric, these are hyper-parameters to be passed at the 
        creation of the `SymbolicAggregateApproximation` object.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.

    Notes
    -----
        The training data are saved to disk if this model is
        serialized and may result in a large model file if the training
        dataset is large.

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
    >>> print(ind)
    [[0]]
    >>> knn2 = KNeighborsTimeSeries(n_neighbors=10,
    ...                             metric="euclidean").fit(time_series)
    >>> print(knn2.kneighbors(return_distance=False))
    [[2 1]
     [2 0]
     [0 1]]
    """  # noqa: E501
    def __init__(self, n_neighbors=5, metric="dtw", metric_params=None,
                 n_jobs=None, verbose=0):
        NearestNeighbors.__init__(self,
                                  n_neighbors=n_neighbors,
                                  algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _is_fitted(self):
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
        else:
            check_is_fitted(self, '_X_fit')

        return True

    def _get_model_params(self):
        if self.metric in TSLEARN_VALID_METRICS:
            return {'_ts_fit': self._ts_fit}
        else:
            return {'_X_fit': self._X_fit}

    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        """
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

        X = check_array(X,
                        allow_nd=True,
                        force_all_finite=(self.metric != "precomputed"))
        X = to_time_series_dataset(X)
        X = check_dims(X)
        if self.metric == "precomputed" and hasattr(self, '_ts_metric'):
            self._ts_fit = X
            self._d = X.shape[2]
            self._X_fit = numpy.zeros((self._ts_fit.shape[0],
                                       self._ts_fit.shape[0]))
        else:
            self._X_fit, self._d = to_sklearn_dataset(X, return_dim=True)
        super().fit(self._X_fit, y)
        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric
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
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

            metric_params = self._get_metric_params()
            check_is_fitted(self, '_ts_fit')
            X = check_array(X, allow_nd=True, force_all_finite=False)
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            if self._ts_metric == "dtw":
                X_ = cdist_dtw(X, self._ts_fit, n_jobs=self.n_jobs,
                               verbose=self.verbose, **metric_params)
            elif self._ts_metric == "ctw":
                X_ = cdist_ctw(X, self._ts_fit, **metric_params)
            elif self._ts_metric == "softdtw":
                X_ = cdist_soft_dtw(X, self._ts_fit, **metric_params)
            else:
                raise ValueError("Invalid metric recorded: %s" %
                                 self._ts_metric)
            pred = KNeighborsTimeSeriesMixin.kneighbors(
                self,
                X=X_,
                n_neighbors=n_neighbors,
                return_distance=return_distance)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            if X is None:
                X_ = None
            else:
                X = check_array(X, allow_nd=True)
                X = to_time_series_dataset(X)
                X_ = to_sklearn_dataset(X)
                X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return KNeighborsTimeSeriesMixin.kneighbors(
                self,
                X=X_,
                n_neighbors=n_neighbors,
                return_distance=return_distance)


class KNeighborsTimeSeriesClassifier(KNeighborsTimeSeriesMixin,
                                     BaseModelPackage,
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
        computations, `n_jobs` and `verbose` keys passed in `metric_params`
        are overridden by the `n_jobs` and `verbose` arguments.
        For 'sax' metric, these are hyper-parameters to be passed at the 
        creation of the `SymbolicAggregateApproximation` object.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    Notes
    -----
        The training data are saved to disk if this model is
        serialized and may result in a large model file if the training
        dataset is large.

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
    >>> clf = KNeighborsTimeSeriesClassifier(n_neighbors=2,
    ...                                      metric="dtw",
    ...                                      metric_params={
    ...                                          "itakura_max_slope": 2.},
    ...                                      n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1]).predict([[1, 2.2, 3.5]])
    array([0])
    """  # noqa: E501
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='dtw',
                 metric_params=None,
                 n_jobs=None,
                 verbose=0):
        KNeighborsClassifier.__init__(self,
                                      n_neighbors=n_neighbors,
                                      weights=weights,
                                      algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _is_fitted(self):
        check_is_fitted(self, '_ts_fit')
        return True

    def _get_model_params(self):
        return {
            '_X_fit': self._X_fit,
            '_ts_fit': self._ts_fit,
            '_d': self._d,
            'classes_': self.classes_,
            '_y': self._y,
            'outputs_2d_': self.outputs_2d_
        }

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, )
            Target values.

        Returns
        -------
        KNeighborsTimeSeriesClassifier
            The fitted estimator
        """
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

        X = check_array(X,
                        allow_nd=True,
                        force_all_finite=(self.metric != "precomputed"))
        X = to_time_series_dataset(X)
        X = check_dims(X)
        if self.metric == "precomputed" and hasattr(self, '_ts_metric'):
            self._ts_fit = X
            if self._ts_metric == 'sax':
                if self.metric_params is not None:
                    self._ts_fit = self._sax_preprocess(X,
                                                        **self.metric_params)
                else:
                    self._ts_fit = self._sax_preprocess(X)

            self._d = X.shape[2]
            self._X_fit = numpy.zeros((self._ts_fit.shape[0],
                                       self._ts_fit.shape[0]))
        else:
            self._X_fit, self._d = to_sklearn_dataset(X, return_dim=True)
        super().fit(self._X_fit, y)
        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric
        return self

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, )
            Array of predicted class labels
        """
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
            X = to_time_series_dataset(X)
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            X_ = self._precompute_cross_dist(X)
            pred = super().predict(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return super().predict(X_)

    def predict_proba(self, X):
        """Predict the class probabilities for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, n_classes)
            Array of predicted class probabilities
        """
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            X_ = self._precompute_cross_dist(X)
            pred = super().predict_proba(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return super().predict_proba(X_)

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}


class KNeighborsTimeSeriesRegressor(KNeighborsTimeSeriesMixin,
                                    KNeighborsRegressor):
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
        computations, `n_jobs` and `verbose` keys passed in `metric_params`
        are overridden by the `n_jobs` and `verbose` arguments.
        For 'sax' metric, these are hyper-parameters to be passed at the 
        creation of the `SymbolicAggregateApproximation` object.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    Examples
    --------
    >>> clf = KNeighborsTimeSeriesRegressor(n_neighbors=2, metric="dtw")
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0.1, 0.1, 1.1]).predict([[1, 2.2, 3.5]])
    array([0.1])
    >>> clf = KNeighborsTimeSeriesRegressor(n_neighbors=2,
    ...                                     metric="dtw",
    ...                                     n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0.1, 0.1, 1.1]).predict([[1, 2.2, 3.5]])
    array([0.1])
    >>> clf = KNeighborsTimeSeriesRegressor(n_neighbors=2,
    ...                                     metric="dtw",
    ...                                     metric_params={
    ...                                         "itakura_max_slope": 2.},
    ...                                     n_jobs=2)
    >>> clf.fit([[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0.1, 0.1, 1.1]).predict([[1, 2.2, 3.5]])
    array([0.1])
    """  # noqa: E501
    def __init__(self,
                 n_neighbors=5,
                 weights='uniform',
                 metric='dtw',
                 metric_params=None,
                 n_jobs=None,
                 verbose=0):
        KNeighborsRegressor.__init__(self,
                                     n_neighbors=n_neighbors,
                                     weights=weights,
                                     algorithm='brute')
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, ) or (n_ts, dim_y)
            Target values.

        Returns
        -------
        KNeighborsTimeSeriesRegressor
            The fitted estimator
        """
        if self.metric in TSLEARN_VALID_METRICS:
            self._ts_metric = self.metric
            self.metric = "precomputed"

        X = check_array(X,
                        allow_nd=True,
                        force_all_finite=(self.metric != "precomputed"))
        X = to_time_series_dataset(X)
        X = check_dims(X)
        if self.metric == "precomputed" and hasattr(self, '_ts_metric'):
            self._ts_fit = X
            self._d = X.shape[2]
            self._X_fit = numpy.zeros((self._ts_fit.shape[0],
                                       self._ts_fit.shape[0]))
        else:
            self._X_fit, self._d = to_sklearn_dataset(X, return_dim=True)
        super().fit(self._X_fit, y)
        if hasattr(self, '_ts_metric'):
            self.metric = self._ts_metric
        return self

    def predict(self, X):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, ) or (n_ts, dim_y)
            Array of predicted targets
        """
        if self.metric in TSLEARN_VALID_METRICS:
            check_is_fitted(self, '_ts_fit')
            X = to_time_series_dataset(X)
            X = check_dims(X, X_fit_dims=self._ts_fit.shape, extend=True,
                           check_n_features_only=True)
            X_ = self._precompute_cross_dist(X)
            pred = super().predict(X_)
            self.metric = self._ts_metric
            return pred
        else:
            check_is_fitted(self, '_X_fit')
            X = check_array(X, allow_nd=True)
            X = to_time_series_dataset(X)
            X_ = to_sklearn_dataset(X)
            X_ = check_dims(X_, X_fit_dims=self._X_fit.shape, extend=False)
            return super().predict(X_)

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}
