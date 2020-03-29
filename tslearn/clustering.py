"""
The :mod:`tslearn.clustering` module gathers time series specific clustering
algorithms.
"""

from __future__ import print_function
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster.k_means_ import _k_init
from sklearn.metrics.cluster import \
    silhouette_score as sklearn_silhouette_score
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist
import numpy
import warnings

from tslearn.metrics import cdist_gak, cdist_dtw, cdist_soft_dtw, \
    cdist_soft_dtw_normalized
from tslearn.barycenters import euclidean_barycenter, \
    dtw_barycenter_averaging, softdtw_barycenter
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import (to_time_series_dataset, to_time_series,
                           ts_size, check_dims)
from tslearn.cycc import cdist_normalized_cc, y_shifted_sbd_vec
from tslearn.bases import BaseModelPackage

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
# Kernel k-means is derived from https://gist.github.com/mblondel/6230787 by
# Mathieu Blondel, under BSD 3 clause license


class EmptyClusterError(Exception):
    def __init__(self, message=""):
        super(EmptyClusterError, self).__init__()
        self.message = message

    def __str__(self):
        if len(self.message) > 0:
            suffix = " (%s)" % self.message
        else:
            suffix = ""
        return "Cluster assignments lead to at least one empty cluster" + \
               suffix


def _check_no_empty_cluster(labels, n_clusters):
    """Check that all clusters have at least one sample assigned.
    """

    for k in range(n_clusters):
        if numpy.sum(labels == k) == 0:
            raise EmptyClusterError


def _check_full_length(centroids):
    """Check that provided centroids are full-length (ie. not padded with
    nans).

    If some centroids are found to be padded with nans, the last value is
    repeated until the end.
    """
    centroids_ = numpy.empty(centroids.shape)
    n, max_sz = centroids.shape[:2]
    for i in range(n):
        sz = ts_size(centroids[i])
        centroids_[i, :sz] = centroids[i, :sz]
        if sz < max_sz:
            centroids_[i, sz:] = centroids[i, sz-1]
    return centroids_


def _compute_inertia(distances, assignments, squared=True):
    """Derive inertia (average of squared distances) from pre-computed
    distances and assignments.

    Examples
    --------
    >>> dists = numpy.array([[1., 2., 0.5], [0., 3., 1.]])
    >>> assign = numpy.array([2, 0])
    >>> _compute_inertia(dists, assign)
    0.125
    """
    n_ts = distances.shape[0]
    if squared:
        return numpy.sum(distances[numpy.arange(n_ts),
                                   assignments] ** 2) / n_ts
    else:
        return numpy.sum(distances[numpy.arange(n_ts), assignments]) / n_ts


def silhouette_score(X, labels, metric=None, sample_size=None,
                     metric_params=None, n_jobs=None, verbose=0, 
                     random_state=None, **kwds):
    """Compute the mean Silhouette Coefficient of all samples (cf.  [1]_ and
    [2]_).

    Read more in the `scikit-learn documentation
    <http://scikit-learn.org/stable/modules/clustering.html\
    #silhouette-coefficient>`_.

    Parameters
    ----------
    X : array [n_ts, n_ts] if metric == "precomputed", or, \
             [n_ts, sz, d] otherwise
        Array of pairwise distances between time series, or a time series
        dataset.
    labels : array, shape = [n_ts]
         Predicted labels for each time series.
    metric : string, callable or None (default: None)
        The metric to use when calculating distance between time series.
        Should be one of {'dtw', 'softdtw', 'euclidean'} or a callable distance
        function or None.
        If 'softdtw' is passed, a normalized version of Soft-DTW is used that
        is defined as `sdtw_(x,y) := sdtw(x,y) - 1/2(sdtw(x,x)+sdtw(y,y))`.
        If X is the distance array itself, use ``metric="precomputed"``.
        If None, dtw is used.
    sample_size : int or None (default: None)
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.
        Value associated to the `"gamma_sdtw"` key corresponds to the gamma
        parameter in Soft-DTW.

        .. deprecated:: 0.2
            `"gamma_sdtw"` as a key for `metric_params` is deprecated in
            version 0.2 and will be removed in 0.4.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    verbose : int (default: 0)
        If nonzero, print information about the inertia while learning
        the model and joblib progress messages are printed.  

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to randomly select a subset of samples.  If int,
        random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`. Used when ``sample_size is not None``.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function,
        just as for the `metric_params` parameter.

    Returns
    -------
    silhouette : float
        Mean Silhouette Coefficient for all samples.

    References
    ----------
    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <http://www.sciencedirect.com/science/article/pii/0377042787901257>`_
    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> from tslearn.metrics import cdist_dtw
    >>> numpy.random.seed(0)
    >>> X = random_walks(n_ts=20, sz=16, d=1)
    >>> labels = numpy.random.randint(2, size=20)
    >>> silhouette_score(X, labels, metric="dtw")  # doctest: +ELLIPSIS
    0.13383800...
    >>> silhouette_score(X, labels, metric="euclidean")  # doctest: +ELLIPSIS
    0.09126917...
    >>> silhouette_score(X, labels, metric="softdtw")  # doctest: +ELLIPSIS
    0.17953934...
    >>> silhouette_score(X, labels, metric="softdtw",
    ...                  metric_params={"gamma": 2.}) \
    # doctest: +ELLIPSIS
    0.17591060...
    >>> silhouette_score(cdist_dtw(X), labels,
    ...                  metric="precomputed")  # doctest: +ELLIPSIS
    0.13383800...
    """
    sklearn_metric = None
    if metric_params is None:
        metric_params_ = {}
    else:
        metric_params_ = metric_params.copy()
    if "gamma_sdtw" in metric_params_.keys():
        warnings.warn(
            "'gamma_sdtw' is deprecated in version 0.2 and will be "
            "removed in 0.4. Use `gamma` instead of `gamma_sdtw` as a "
            "`metric_params` key to remove this warning.",
            DeprecationWarning, stacklevel=2)
        metric_params_["gamma"] = metric_params_["gamma_sdtw"]
        del metric_params_["gamma_sdtw"]
    for k in kwds.keys():
        metric_params_[k] = kwds[k]
    if "n_jobs" in metric_params_.keys():
        del metric_params_["n_jobs"]
    if metric == "precomputed":
        sklearn_X = X
    elif metric == "dtw" or metric is None:
        sklearn_X = cdist_dtw(X, n_jobs=n_jobs, verbose=verbose,
                              **metric_params_)
    elif metric == "softdtw":
        sklearn_X = cdist_soft_dtw_normalized(X, **metric_params_)
    elif metric == "euclidean":
        X_ = to_time_series_dataset(X)
        X_ = X_.reshape((X.shape[0], -1))
        sklearn_X = cdist(X_, X_, metric="euclidean")
    else:
        X_ = to_time_series_dataset(X)
        n, sz, d = X_.shape
        sklearn_X = X_.reshape((n, -1))

        def sklearn_metric(x, y):
            return metric(to_time_series(x.reshape((sz, d)),
                                         remove_nans=True),
                          to_time_series(y.reshape((sz, d)),
                                         remove_nans=True))
    metric = "precomputed" if sklearn_metric is None else sklearn_metric
    return sklearn_silhouette_score(X=sklearn_X,
                                    labels=labels,
                                    metric=metric,
                                    sample_size=sample_size,
                                    random_state=random_state,
                                    **kwds)


def _check_initial_guess(init, n_clusters):
    if hasattr(init, '__array__'):
        assert init.shape[0] == n_clusters, \
            "Initial guess index array must contain {} samples," \
            " {} given".format(n_clusters, init.shape[0])


class GlobalAlignmentKernelKMeans(BaseEstimator, BaseModelPackage,
                                  ClusterMixin):
    """Global Alignment Kernel K-means.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than
        this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm
        stops.

    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.

    sigma : float or "auto" (default: "auto")
        Bandwidth parameter for the Global Alignment kernel. If set to 'auto',
        it is computed based on a sampling of the training set
        (cf :ref:`tslearn.metrics.sigma_gak <fun-tslearn.metrics.sigma_gak>`)

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for GAK cross-similarity matrix
        computations.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    verbose : int (default: 0)
        If nonzero, joblib progress messages are printed.  

    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it
        fixes the seed. Defaults to the global
        numpy random number generator.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center (computed
        using the kernel trick).

    sample_weight_ : numpy.ndarray
        The weight given to each sample from the data provided to fit.

    n_iter_ : int
        The number of iterations performed during fit.

    Notes
    -----
        The training data are saved to disk if this model is
        serialized and may result in a large model file if the training
        dataset is large.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> gak_km = GlobalAlignmentKernelKMeans(n_clusters=3,
    ...                                      random_state=0).fit(X)

    References
    ----------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.

    Fast Global Alignment Kernels.
    Marco Cuturi.
    ICML 2011.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-6, n_init=1, sigma=1.,
                 n_jobs=None, verbose=0, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.sigma = sigma
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _is_fitted(self):
        check_is_fitted(self, '_X_fit')
        return True

    def _get_model_params(self):
        return {
            '_X_fit': self._X_fit,
            'sample_weight_': self.sample_weight_,
            'labels_': self.labels_
        }

    def _get_kernel(self, X, Y=None):
        return cdist_gak(X, Y, sigma=self.sigma, n_jobs=self.n_jobs,
                         verbose=self.verbose)

    def _fit_one_init(self, K, rs):
        n_samples = K.shape[0]

        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = numpy.empty((n_samples, self.n_clusters))
        old_inertia = numpy.inf

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist)
            self.labels_ = dist.argmin(axis=1)
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            self.inertia_ = self._compute_inertia(dist)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")

            if numpy.abs(old_inertia - self.inertia_) < self.tol:
                break
            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self

    def fit(self, X, y=None, sample_weight=None):
        """Compute kernel k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        y
            Ignored
        sample_weight : array-like of shape=(n_ts, ) or None (default: None)
            Weights to be given to time series in the learning process. By
            default, all time series weights are equal.
        """

        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X, X_fit=None)

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)

        max_attempts = max(self.n_init, 10)

        self.labels_ = None
        self.inertia_ = None
        self.sample_weight_ = None
        self._X_fit = None
        # n_iter_ will contain the number of iterations the most
        # successful run required.
        self.n_iter_ = 0

        n_samples = X.shape[0]
        K = self._get_kernel(X)
        sw = (sample_weight if sample_weight is not None
              else numpy.ones(n_samples))
        self.sample_weight_ = sw
        rs = check_random_state(self.random_state)

        last_correct_labels = None
        min_inertia = numpy.inf
        n_attempts = 0
        n_successful = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(K, rs)
                if self.inertia_ < min_inertia:
                    last_correct_labels = self.labels_
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        if n_successful > 0:
            self.labels_ = last_correct_labels
            self.inertia_ = min_inertia
            self._X_fit = X
        return self

    def _compute_dist(self, K, dist):
        """Compute a n_samples x n_clusters distance matrix using the kernel
        trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = (self.labels_ == j)

            if numpy.sum(mask) == 0:
                raise EmptyClusterError("try smaller n_cluster or better "
                                        "kernel parameters")

            # NB: we use a normalized kernel so k(x,x) = 1 for all x
            # (including the centroid)
            dist[:, j] = 2 - 2 * numpy.sum(sw[mask] * K[:, mask],
                                           axis=1) / sw[mask].sum()

    @staticmethod
    def _compute_inertia(dist_sq):
        return dist_sq.min(axis=1).sum()

    def fit_predict(self, X, y=None):
        """Fit kernel k-means clustering using X and then predict the closest
        cluster each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit
        and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        y
            Ignored

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        check_is_fitted(self, '_X_fit')
        X = check_dims(X, self._X_fit)
        K = self._get_kernel(X, self._X_fit)
        n_samples = X.shape[0]
        dist = numpy.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist)
        return dist.argmin(axis=1)

    def _get_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}


class TimeSeriesCentroidBasedClusteringMixin:
    """Mixin class for centroid-based clustering of time series."""
    def _post_fit(self, X_fitted, centroids, inertia):
        if numpy.isfinite(inertia) and (centroids is not None):
            self.cluster_centers_ = centroids
            self._assign(X_fitted)
            self._X_fit = X_fitted
            self.inertia_ = inertia
        else:
            self._X_fit = None


class TimeSeriesKMeans(BaseEstimator, BaseModelPackage, ClusterMixin,
                       TimeSeriesCentroidBasedClusteringMixin):
    """K-means clustering for time-series data.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.

    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than
        this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm
        stops.

    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia.

    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric to be used for both cluster assignment and barycenter
        computation. If "dtw", DBA is used for barycenter
        computation.

    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process. Only used
        if `metric="dtw"` or `metric="softdtw"`.

    metric_params : dict or None (default: None)
        Parameter values for the chosen metric.
        For metrics that accept parallelization of the cross-distance matrix
        computations, `n_jobs` key passed in `metric_params` is overridden by
        the `n_jobs` argument.
        Value associated to the `"gamma_sdtw"` key corresponds to the gamma
        parameter in Soft-DTW.

        .. deprecated:: 0.2
            `"gamma_sdtw"` as a key for `metric_params` is deprecated in
            version 0.2 and will be removed in 0.4.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for cross-distance matrix
        computations.
        Ignored if the cross-distance matrix cannot be computed using
        parallelization.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    dtw_inertia: bool (default: False)
        Whether to compute DTW inertia even if DTW is not the chosen metric.

    verbose : int (default: 0)
        If nonzero, print information about the inertia while learning
        the model and joblib progress messages are printed.  

    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it
        fixes the seed. Defaults to the global
        numpy random number generator.

    init : {'k-means++', 'random' or an ndarray} (default: 'k-means++')
        Method for initialization:
        'k-means++' : use k-means++ heuristic. See `scikit-learn's k_init_
        <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/\
        cluster/k_means_.py>`_ for more.
        'random': choose k observations (rows) at random from data for the
        initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, ts_size, d)
        and gives the initial centers.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point.

    cluster_centers_ : numpy.ndarray
        Cluster centers.

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    n_iter_ : int
        The number of iterations performed during fit.

    Notes
    -----
        If `metric` is set to `"euclidean"`, the algorithm expects a dataset of
        equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5,
    ...                       random_state=0).fit(X)
    >>> km.cluster_centers_.shape
    (3, 32, 1)
    >>> km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5,
    ...                           max_iter_barycenter=5,
    ...                           random_state=0).fit(X)
    >>> km_dba.cluster_centers_.shape
    (3, 32, 1)
    >>> km_sdtw = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5,
    ...                            max_iter_barycenter=5,
    ...                            metric_params={"gamma": .5},
    ...                            random_state=0).fit(X)
    >>> km_sdtw.cluster_centers_.shape
    (3, 32, 1)
    >>> X_bis = to_time_series_dataset([[1, 2, 3, 4],
    ...                                 [1, 2, 3],
    ...                                 [2, 5, 6, 7, 8, 9]])
    >>> km = TimeSeriesKMeans(n_clusters=2, max_iter=5,
    ...                       metric="dtw", random_state=0).fit(X_bis)
    >>> km.cluster_centers_.shape
    (2, 3, 1)
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-6, n_init=1,
                 metric="euclidean", max_iter_barycenter=100,
                 metric_params=None, n_jobs=None, dtw_inertia=False,
                 verbose=0, random_state=None, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.metric = metric
        self.max_iter_barycenter = max_iter_barycenter
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.dtw_inertia = dtw_inertia
        self.verbose = verbose
        self.random_state = random_state
        self.init = init

        if self.metric_params is not None and \
            "gamma_sdtw" in self.metric_params.keys():
            warnings.warn(
                "'gamma_sdtw' is deprecated in version 0.2 and will be "
                "removed in 0.4. Use `gamma` instead of `gamma_sdtw` as a "
                "`metric_params` key to remove this warning.",
                DeprecationWarning, stacklevel=2)

    def _is_fitted(self):
        check_is_fitted(self, ['cluster_centers_'])
        return True

    def _get_model_params(self):
        return {'cluster_centers_': self.cluster_centers_}

    def _fit_one_init(self, X, x_squared_norms, rs):
        n_ts, _, d = X.shape
        sz = min([ts_size(ts) for ts in X])
        if hasattr(self.init, '__array__'):
            self.cluster_centers_ = self.init.copy()
        elif self.init == "k-means++":
            self.cluster_centers_ = _k_init(X[:, :sz, :].reshape((n_ts, -1)),
                                            self.n_clusters,
                                            x_squared_norms,
                                            rs).reshape((-1, sz, d))
        elif self.init == "random":
            indices = rs.choice(X.shape[0], self.n_clusters)
            self.cluster_centers_ = X[indices].copy()
        else:
            raise ValueError("Value %r for parameter 'init'"
                             "is invalid" % self.init)
        self.cluster_centers_ = _check_full_length(self.cluster_centers_)
        old_inertia = numpy.inf

        for it in range(self.max_iter):
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")
            self._update_centroids(X)

            if numpy.abs(old_inertia - self.inertia_) < self.tol:
                break
            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self

    def _assign(self, X, update_class_attributes=True):
        if self.metric_params is None:
            metric_params = {}
        else:
            metric_params = self.metric_params.copy()
        if "gamma_sdtw" in metric_params.keys():
            metric_params["gamma"] = metric_params["gamma_sdtw"]
            del metric_params["gamma_sdtw"]
        if "n_jobs" in metric_params.keys():
            del metric_params["n_jobs"]
        if self.metric == "euclidean":
            dists = cdist(X.reshape((X.shape[0], -1)),
                          self.cluster_centers_.reshape((self.n_clusters, -1)),
                          metric="euclidean")
        elif self.metric == "dtw":
            dists = cdist_dtw(X, self.cluster_centers_, n_jobs=self.n_jobs,
                              verbose=self.verbose, **metric_params)
        elif self.metric == "softdtw":
            dists = cdist_soft_dtw(X, self.cluster_centers_, **metric_params)
        else:
            raise ValueError("Incorrect metric: %s (should be one of 'dtw', "
                             "'softdtw', 'euclidean')" % self.metric)
        matched_labels = dists.argmin(axis=1)
        if update_class_attributes:
            self.labels_ = matched_labels
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            if self.dtw_inertia and self.metric != "dtw":
                inertia_dists = cdist_dtw(X, self.cluster_centers_,
                                          n_jobs=self.n_jobs,
                                          verbose=self.verbose)
            else:
                inertia_dists = dists
            self.inertia_ = _compute_inertia(inertia_dists,
                                             self.labels_,
                                             self._squared_inertia)
        return matched_labels

    def _update_centroids(self, X):
        if self.metric_params is None:
            metric_params = {}
        else:
            metric_params = self.metric_params.copy()
        if "gamma_sdtw" in metric_params.keys():
            metric_params["gamma"] = metric_params["gamma_sdtw"]
            del metric_params["gamma_sdtw"]
        for k in range(self.n_clusters):
            if self.metric == "dtw":
                self.cluster_centers_[k] = dtw_barycenter_averaging(
                    X=X[self.labels_ == k],
                    barycenter_size=None,
                    init_barycenter=self.cluster_centers_[k],
                    metric_params=metric_params,
                    verbose=False)
            elif self.metric == "softdtw":
                self.cluster_centers_[k] = softdtw_barycenter(
                    X=X[self.labels_ == k],
                    max_iter=self.max_iter_barycenter,
                    init=self.cluster_centers_[k],
                    **metric_params)
            else:
                self.cluster_centers_[k] = euclidean_barycenter(
                    X=X[self.labels_ == k])

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        y
            Ignored
        """

        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')

        self.labels_ = None
        self.inertia_ = numpy.inf
        self.cluster_centers_ = None
        self._X_fit = None
        self._squared_inertia = True

        self.n_iter_ = 0

        max_attempts = max(self.n_init, 10)

        X_ = to_time_series_dataset(X)
        rs = check_random_state(self.random_state)
        x_squared_norms = cdist(X_.reshape((X_.shape[0], -1)),
                                numpy.zeros((1, X_.shape[1] * X_.shape[2])),
                                metric="sqeuclidean").reshape((1, -1))
        _check_initial_guess(self.init, self.n_clusters)

        best_correct_centroids = None
        min_inertia = numpy.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(X_, x_squared_norms, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self

    def fit_predict(self, X, y=None):
        """Fit k-means clustering using X and then predict the closest cluster
        each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit
        and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        y
            Ignored

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        return self.fit(X, y).labels_

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, allow_nd=True, force_all_finite='allow-nan')
        check_is_fitted(self, 'cluster_centers_')
        X = check_dims(X, self.cluster_centers_)
        X_ = to_time_series_dataset(X)
        return self._assign(X_, update_class_attributes=False)

    def _get_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}


class KShape(BaseEstimator, BaseModelPackage, ClusterMixin,
             TimeSeriesCentroidBasedClusteringMixin):
    """KShape clustering for time series.

    KShape was originally presented in [1]_.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 100)
        Maximum number of iterations of the k-Shape algorithm.

    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than
        this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm
        stops.

    n_init : int (default: 1)
        Number of time the k-Shape algorithm will be run with different
        centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.

    verbose : bool (default: False)
        Whether or not to print information about the inertia while learning
        the model.

    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it
        fixes the seed. Defaults to the global
        numpy random number generator.

    init : {'random' or ndarray} (default: 'random')
        Method for initialization.
        'random': choose k observations (rows) at random from data for the
        initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, ts_size, d)
        and gives the initial centers.

    Attributes
    ----------
    cluster_centers_ : numpy.ndarray of shape (sz, d).
        Centroids

    labels_ : numpy.ndarray of integers with shape (n_ts, ).
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    n_iter_ : int
        The number of iterations performed during fit.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
    >>> ks = KShape(n_clusters=3, n_init=1, random_state=0).fit(X)
    >>> ks.cluster_centers_.shape
    (3, 32, 1)

    References
    ----------
    .. [1] J. Paparrizos & L. Gravano. k-Shape: Efficient and Accurate
       Clustering of Time Series. SIGMOD 2015. pp. 1855-1870.
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-6, n_init=1,
                 verbose=False, random_state=None, init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.verbose = verbose
        self.init = init

    def _get_model_params(self):
        """
        Get the model parameters

        Returns
        -------
        params : dict
            Model parameters (attributes) that are sufficient
            to recapitulate the model
        """

        return {'cluster_centers_': self.cluster_centers_,
                'norms_': self.norms_,
                'norms_centroids_': self.norms_centroids_,
                }

    def _is_fitted(self):
        """
        Check if the model has been fit.

        Returns
        -------
        bool
        """

        check_is_fitted(self,
                        ['cluster_centers_', 'norms_', 'norms_centroids_'])
        return True

    def _shape_extraction(self, X, k):
        sz = X.shape[1]
        Xp = y_shifted_sbd_vec(self.cluster_centers_[k], X[self.labels_ == k],
                               norm_ref=-1,
                               norms_dataset=self.norms_[self.labels_ == k])
        S = numpy.dot(Xp[:, :, 0].T, Xp[:, :, 0])
        Q = numpy.eye(sz) - numpy.ones((sz, sz)) / sz
        M = numpy.dot(Q.T, numpy.dot(S, Q))
        _, vec = numpy.linalg.eigh(M)
        mu_k = vec[:, -1].reshape((sz, 1))

        # The way the optimization problem is (ill-)formulated, both mu_k and
        # -mu_k are candidates for barycenters
        # In the following, we check which one is best candidate
        dist_plus_mu = numpy.sum(numpy.linalg.norm(Xp - mu_k, axis=(1, 2)))
        dist_minus_mu = numpy.sum(numpy.linalg.norm(Xp + mu_k, axis=(1, 2)))
        if dist_minus_mu < dist_plus_mu:
            mu_k *= -1

        return mu_k

    def _update_centroids(self, X):
        for k in range(self.n_clusters):
            self.cluster_centers_[k] = self._shape_extraction(X, k)
        self.cluster_centers_ = TimeSeriesScalerMeanVariance(
            mu=0., std=1.).fit_transform(self.cluster_centers_)
        self.norms_centroids_ = numpy.linalg.norm(self.cluster_centers_,
                                                  axis=(1, 2))

    def _cross_dists(self, X):
        return 1. - cdist_normalized_cc(X, self.cluster_centers_,
                                        norms1=self.norms_,
                                        norms2=self.norms_centroids_,
                                        self_similarity=False)

    def _assign(self, X):
        dists = self._cross_dists(X)
        self.labels_ = dists.argmin(axis=1)
        _check_no_empty_cluster(self.labels_, self.n_clusters)
        self.inertia_ = _compute_inertia(dists, self.labels_)

    def _fit_one_init(self, X, rs):
        if hasattr(self.init, '__array__'):
            self.cluster_centers_ = self.init.copy()
        elif self.init == "random":
            indices = rs.choice(X.shape[0], self.n_clusters)
            self.cluster_centers_ = X[indices].copy()
        else:
            raise ValueError("Value %r for parameter 'init' is "
                             "invalid" % self.init)
        self.norms_centroids_ = numpy.linalg.norm(self.cluster_centers_,
                                                  axis=(1, 2))
        self._assign(X)
        old_inertia = numpy.inf

        for it in range(self.max_iter):
            old_cluster_centers = self.cluster_centers_.copy()
            self._update_centroids(X)
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")

            if numpy.abs(old_inertia - self.inertia_) < self.tol or \
                    (old_inertia - self.inertia_ < 0):
                self.cluster_centers_ = old_cluster_centers
                self._assign(X)
                break

            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self

    def fit(self, X, y=None):
        """Compute k-Shape clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        y
            Ignored
        """
        X = check_array(X, allow_nd=True)

        max_attempts = max(self.n_init, 10)

        self.labels_ = None
        self.inertia_ = numpy.inf
        self.cluster_centers_ = None

        self.norms_ = 0.
        self.norms_centroids_ = 0.

        self.n_iter_ = 0

        X_ = to_time_series_dataset(X)
        self._X_fit = X_
        self.norms_ = numpy.linalg.norm(X_, axis=(1, 2))

        _check_initial_guess(self.init, self.n_clusters)

        rs = check_random_state(self.random_state)

        best_correct_centroids = None
        min_inertia = numpy.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(X_, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self.norms_centroids_ = numpy.linalg.norm(self.cluster_centers_,
                                                  axis=(1, 2))
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self

    def fit_predict(self, X, y=None):
        """Fit k-Shape clustering using X and then predict the closest cluster
        each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit
        and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        y
            Ignored

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, allow_nd=True)
        check_is_fitted(self,
                        ['cluster_centers_', 'norms_', 'norms_centroids_'])

        X_ = to_time_series_dataset(X)

        X = check_dims(X, self.cluster_centers_)

        X_ = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_)
        dists = self._cross_dists(X_)
        return dists.argmin(axis=1)
