"""DBSCAN clustering."""

import copy

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesMixin, BaseModelPackage
from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.metrics import TSLEARN_VALID_METRICS, METRIC_TO_FUNCTION, VARIABLE_LENGTH_METRICS
from tslearn.utils import check_array, to_time_series_dataset, check_dims, to_sklearn_dataset
from tslearn.neighbors import KNeighborsTimeSeries

from sklearn.cluster._dbscan_inner import dbscan_inner


class TimeSeriesDBSCAN(TimeSeriesMixin, ClusterMixin, BaseEstimator, BaseModelPackage):
    """
    DBSCAN clustering for time series.

    Parameters
    ----------
    eps : float (default: 0.5)
        The maximum distance between two time series for one to be considered
        as in the neighborhood of the other.
    min_ts : int (default: 5)
        The number of time series (including itself) in a neighborhood for a time series
        to be considered as a core point.
    metric: {'dtw', 'ctw', 'frechet', 'euclidean', 'precomputed'} (default: 'dtw')
        Metric to be used for similarity measure between time series.
    metric_params : dict (default: None)
        Additional keyword arguments to pass to the metric function.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.
    n_jobs : int or None (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_
        for more details.

    Attributes
    ----------
    core_ts_indices_ : numpy.ndarray of shape (n_core_ts).
        Indices of core time series.

    components_: numpy.ndarray of shape (n_core_ts, sz, d)
        Copy of each core time series found by training.

    labels_ : numpy.ndarray of integers with shape (n_ts).
        Labels of each time series. Noisy time series are given the label -1.

    n_features_in_ : int
        Number of features seen during training.

    Notes
    -----
        If `metric` is set to `"euclidean"`, the algorithm expects a dataset of
        equal-sized time series.

     Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    >>> X, y = random_walk_blobs(n_ts_per_blob=20, sz=32, d=2, n_blobs=4, random_state=0)
    >>> X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
    >>> db = TimeSeriesDBSCAN(eps=4, min_ts=3).fit(X)
    >>> np.unique(db.labels_) # Clusters and noise
    array([-1,  0,  1,  2,  3])
    >>> list(db.labels_).count(-1) # Nb noisy elements
    37
    """

    VALID_METRICS = set(("dtw", "ctw", "frechet", "euclidean", "precomputed"))

    def __init__(
            self,
            eps=0.5,
            min_ts=5,
            metric='dtw',
            metric_params=None,
            n_jobs=None
    ):
        self.eps = eps
        self.min_ts = min_ts
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def _is_fitted(self):
        return check_is_fitted(self) or True # pragma: no cover

    def _get_metric_params(self):
        if self.metric_params is None:
            metric_params = {}
        else:
            metric_params = copy.deepcopy(self.metric_params)
        if self.n_jobs is not None:
            metric_params.update({"n_jobs": self.n_jobs})
        return metric_params

    def fit(self, X, y=None):
        """Compute DBSCAN clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        y
            Ignored

        Returns
        -------
        TimeSeriesDBSCAN
            The fitted estimator
        """
        if self.metric not in self.VALID_METRICS:
            raise ValueError("Metric must be one of: {}".format(self.VALID_METRICS))

        X = check_array(
            X,
            allow_nd=True,
            # For variable length time series with dedicated metric
            force_all_finite="allow-nan" if self.metric in TSLEARN_VALID_METRICS else True
        )
        X = to_time_series_dataset(X)
        X = check_dims(X)

        neighbors_model = KNeighborsTimeSeries(
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        neighbors_model.fit(X)

        if self.metric in TSLEARN_VALID_METRICS:
            distance_matrix = METRIC_TO_FUNCTION[self.metric](
                X,
                **self._get_metric_params()
            )
            with neighbors_model._patch_attribute("metric", "precomputed"):
                neighborhoods = neighbors_model.radius_neighbors(
                    distance_matrix,
                    radius=self.eps,
                    return_distance=False
                )
        else:
            neighborhoods = neighbors_model.radius_neighbors(
                to_sklearn_dataset(X),
                radius=self.eps,
                return_distance=False
            )

        n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_ts, dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)

        self.core_ts_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_ts_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_ts_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))

        self.n_features_in_ = X.shape[-1]
        return self

    def fit_predict(self, X, y=None):
        """Compute DBSCAN clustering.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : array of shape=(n_ts)
            Index of the cluster each TS belongs to. Noisy TS are given the label -1.
        """
        self.fit(X)
        return self.labels_

    def _more_tags(self):
        tags = super()._more_tags()
        tags.update({
            "allow_nan": self.metric in VARIABLE_LENGTH_METRICS,
            ALLOW_VARIABLE_LENGTH: self.metric in VARIABLE_LENGTH_METRICS}
        )
        return tags

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self.metric in VARIABLE_LENGTH_METRICS
        tags.allow_variable_length = self.metric in VARIABLE_LENGTH_METRICS
        return tags
