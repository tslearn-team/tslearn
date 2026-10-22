import numpy
import numpy as np
from sklearn.exceptions import NotFittedError

from sklearn.neighbors import (KNeighborsClassifier, NearestNeighbors,
                               KNeighborsRegressor)
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import BaseModelPackage, TimeSeriesMixin
from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.metrics import (
    _cdist_dtw,
    _cdist_ctw,
    _cdist_soft_dtw,
    _cdist_sax,
    _cdist_frechet,
    TSLEARN_VALID_METRICS
)
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.utils import (
    to_time_series_dataset,
    to_sklearn_dataset,
    check_array,
    check_dims
)


class KNeighborsTimeSeriesMixin(TimeSeriesMixin):
    """Mixin for k-neighbors searches on Time Series."""

    def _fit_(self, X, y):
        X = check_array(X,
                        allow_nd=True,
                        force_all_finite="allow-nan" if self.metric in TSLEARN_VALID_METRICS else True)
        X = to_time_series_dataset(X)
        X = check_dims(X)
        self._X_fit_dims_ = X.shape
        if self.metric in TSLEARN_VALID_METRICS:
            # Allow nans
            with self._patch_attribute("metric", "nan_euclidean"):
                super().fit(to_sklearn_dataset(X), y)
                self.effective_metric_ = self.metric
        else:
            super().fit(to_sklearn_dataset(X), y)
        self.n_features_in_ = self._X_fit_dims_[-1]

        return self

    def _is_fitted(self):
        try:
            check_is_fitted(self)
        except NotFittedError:
            return False
        return True

    def _sax_preprocess(self, X, n_segments=10, alphabet_size_avg=4, scale=False):
            if not hasattr(self, '_sax') or self._sax is None:
                self._sax = SymbolicAggregateApproximation(
                    n_segments=n_segments,
                    alphabet_size_avg=alphabet_size_avg,
                    scale=scale
                ).fit(X)
            return self._sax.transform(X), self._sax.breakpoints_avg_

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

        metric_params = self._get_metric_params()

        if self.metric == "dtw":
            return _cdist_dtw(
                X,
                other_X,
                n_jobs=self.n_jobs,
                **metric_params
            )
        elif self.metric == "ctw":
            return _cdist_ctw(
                X,
                other_X,
                **metric_params)
        elif self.metric == "softdtw":
            return _cdist_soft_dtw(
                X,
                other_X,
                n_jobs=self.n_jobs,
                **metric_params
            )
        elif self.metric == "sax":
            X_sax, breakpoint_avg = self._sax_preprocess(X, **metric_params)
            other_X_sax = None
            if other_X is not None:
                other_X_sax, _ = self._sax_preprocess(other_X, **metric_params)
            return _cdist_sax(
                X_sax,
                breakpoint_avg,
                X.shape[1],
                other_X_sax,
                n_jobs=self.n_jobs
            )
        elif self.metric == "frechet":
            return _cdist_frechet(
                X,
                other_X,
                n_jobs=self.n_jobs,
                **metric_params
            )
        else:
            raise ValueError("Invalid metric recorded: %s" %
                             self.metric)

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

        if self.metric not in TSLEARN_VALID_METRICS:
            X = to_sklearn_dataset(X) if X is not None else X
            with self._patch_attribute("n_features_in_", self._X_fit_dims_[1] * self._X_fit_dims_[2]):
                return super().kneighbors(X, n_neighbors, return_distance)

        self_neighbors = X is None

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if self._X_fit_dims_[0] <  n_neighbors + int(self_neighbors):
            raise ValueError(f"Expected n_neighbors <{'' if self_neighbors else '='} number of fitted TS")

        fitted = self._fit_X.reshape(self._X_fit_dims_)

        full_dist_matrix = self._precompute_cross_dist(fitted, X).T

        # Code similar to sklearn (sklearn/neighbors/base.py), to make sure
        # that TimeSeriesKNeighbor~(metric='euclidean') has the same results as
        # feeding a distance matrix to sklearn.KNeighbors~(metric='euclidean')
        kbin = min(n_neighbors - 1 + int(self_neighbors),
                   full_dist_matrix.shape[1] - 1)
        # argpartition will make sure the first `kbin` entries are the
        # `kbin` smallest ones (but in arbitrary order) --> complexity: O(n)
        ind = numpy.argpartition(full_dist_matrix, kbin, axis=1)

        if self_neighbors:
            keep = ind != numpy.arange(ind.shape[0])[:, None]
            ind = ind[keep].reshape(ind.shape[0], -1)
        ind = ind[:, :n_neighbors]

        sample_range = numpy.arange(full_dist_matrix.shape[0])[:, None]
        # Sort the `kbin` nearest neighbors according to distance
        ind = ind[
            sample_range, numpy.argsort(full_dist_matrix[sample_range, ind])]

        if return_distance:
            dist = full_dist_matrix[sample_range, ind]
            return dist, ind
        else:
            return ind

    def _get_model_params(self):
        params = super()._get_model_params()
        params.update({"_X_fit_dims_": numpy.array(self._X_fit_dims_)})
        params["_fit_X"] = self._fit_X
        params["_fit_method"] = self._fit_method
        if hasattr(self, "_y"):
            params["_y"] = self._y
        return params

    @staticmethod
    def _organize_model(cls, model):
        inst = super()._organize_model(cls, model)
        if not hasattr(inst, "effective_metric_params_"):
            inst.effective_metric_params_ = {}
        return inst

    def _more_tags(self):
        tags = super()._more_tags()
        tags.update({
            'allow_nan': self.metric in (TSLEARN_VALID_METRICS + ["nan_euclidean"]),
            ALLOW_VARIABLE_LENGTH: self.metric in TSLEARN_VALID_METRICS}
        )
        return tags

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self.metric in (TSLEARN_VALID_METRICS + ["nan_euclidean"])
        tags.allow_variable_length = self.metric in TSLEARN_VALID_METRICS
        return tags


def _predict_generic(caller, X, predict_func):
    """Predict the class labels or target (depending on predict_func) for the provided data

    Parameters
    ----------
    X : array-like, shape (n_ts, sz, d)
        Test samples.

    Returns
    -------
    array, shape = (n_ts, ) or (n_ts, dim_y)
        Returns the result of predict_func 
    """
    check_is_fitted(caller)

    if X is not None:
        X = check_array(
            X,
            allow_nd=True,
            force_all_finite="allow-nan" if caller.metric in TSLEARN_VALID_METRICS else True
        )
        X = to_time_series_dataset(X)
        X = check_dims(
            X,
            X_fit_dims=tuple(caller._X_fit_dims_),
            extend=True,
            check_n_features_only=caller.metric in TSLEARN_VALID_METRICS
        )
    return predict_func(X)


class KNeighborsTimeSeries(KNeighborsTimeSeriesMixin,
                           NearestNeighbors,
                           BaseModelPackage):
    """Unsupervised learner for implementing neighbor searches for Time Series.

    Parameters
    ----------
    n_neighbors : int (default: 5)
        Number of nearest neighbors to be considered for the decision.

    metric : {'dtw', 'softdtw', 'ctw', 'euclidean', 'sqeuclidean', \
              'cityblock', 'sax'} (default: 'dtw')
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
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
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

    def fit(self, X, y=None):
        """Fit the model using X as training data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        """
        return super()._fit_(X, y)

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            The query time series dataset.
            If not provided, neighbors of each indexed TS are returned.
            In this case, the query TS is not considered its own neighbor.
        n_neighbors : int or None (default: None)
            Number of neighbors to get (default is the value passed to the
            constructor).
        return_distance : boolean (default: True).
            If False, distances will not be returned

        Returns
        -------
        dist : array
            Array representing the distance to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        """
        check_is_fitted(self)

        if X is not None:
            X = check_array(X, allow_nd=True, force_all_finite=self.metric in TSLEARN_VALID_METRICS)
            X = check_dims(
                X,
                X_fit_dims=tuple(self._X_fit_dims_),
                extend=True,
                check_n_features_only=self.metric in TSLEARN_VALID_METRICS
            )
        return KNeighborsTimeSeriesMixin.kneighbors(
            self,
            X=X,
            n_neighbors=n_neighbors,
            return_distance=return_distance
        )

    def radius_neighbors(
        self, X=None, radius=1, return_distance=True, sort_results=False
    ):
        """
        Find the neighbors within a given radius of a point or points.

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            The query time series dataset.
            If not provided, neighbors of each indexed TS are returned.
            In this case, the query TS is not considered its own neighbor.

        radius : float (default: 1.0).
            Limiting distance of neighbors to return.

        return_distance : boolean (default: True).
            If False, distances will not be returned

        sort_results : boolean (default: False).
            If True, the distances and indices will be sorted by increasing distances before being returned.
            If False, the results may not be sorted.
            If return_distance=False, setting sort_results=True will result in an error.

        Returns
        -------
            dist : ndarray of shape (n_ts,) of arrays
                Array representing the distance to each TS, only present if return_distance=True
            ind : ndarray of shape (n_ts,) of arrays
                Indices of the TS that lie within `radius`
        """
        check_is_fitted(self)

        if self.metric=="softdtw":
            raise ValueError("Not implemented for softdtw")

        if X is not None:
            X = check_array(X, allow_nd=True, force_all_finite=self.metric in TSLEARN_VALID_METRICS)
            X = check_dims(
                X,
                X_fit_dims=tuple(self._X_fit_dims_),
                extend=True,
                check_n_features_only=self.metric in TSLEARN_VALID_METRICS
            )
        if self.metric in TSLEARN_VALID_METRICS:
            fitted = self._fit_X.reshape(self._X_fit_dims_)
            full_dist_matrix = self._precompute_cross_dist(fitted, X).T
            if X is None:
                # Ensure not a neighbor of itself
                full_dist_matrix[np.diag_indices_from(full_dist_matrix)] = radius + int(radius >= 0)
            with self._patch_attribute("metric", "precomputed"):
                with self._patch_attribute("effective_metric_", "precomputed"):
                    return super().radius_neighbors(
                        full_dist_matrix, radius, return_distance, sort_results
                    )

        else:
            X = to_sklearn_dataset(X) if X is not None else X
            with self._patch_attribute("n_features_in_", self._X_fit_dims_[1] * self._X_fit_dims_[2]):
                return super().radius_neighbors(X, radius, return_distance, sort_results)


class KNeighborsTimeSeriesClassifier(KNeighborsTimeSeriesMixin,
                                     KNeighborsClassifier,
                                     BaseModelPackage):
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
    metric : one of the metrics allowed for :class:`.KNeighborsTimeSeries`\
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
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
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
        return super()._fit_(X, y)

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
        return _predict_generic(self, X, super().predict)

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
        return _predict_generic(self, X, super().predict_proba)


class KNeighborsTimeSeriesRegressor(KNeighborsTimeSeriesMixin,
                                    KNeighborsRegressor,
                                    BaseModelPackage):
    """Regressor implementing the k-nearest neighbors vote for Time Series.

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
    metric : one of the metrics allowed for :class:`.KNeighborsTimeSeries`\
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
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
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
        return super()._fit_(X, y)

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
        return _predict_generic(self, X, super().predict)

