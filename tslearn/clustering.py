"""
The :mod:`tslearn.clustering` module gathers time series specific clustering algorithms.
"""

from __future__ import print_function
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster.k_means_ import _k_init
from sklearn.metrics.cluster import silhouette_score as sklearn_silhouette_score
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist
import numpy

from tslearn.metrics import cdist_gak, cdist_dtw, cdist_soft_dtw, dtw
from tslearn.barycenters import EuclideanBarycenter, DTWBarycenterAveraging, SoftDTWBarycenter
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset, to_time_series, ts_size
from tslearn.cycc import cdist_normalized_cc, y_shifted_sbd_vec


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
# Kernel k-means is derived from https://gist.github.com/mblondel/6230787 by Mathieu Blondel, under BSD 3 clause license


class EmptyClusterError(Exception):
    def __init__(self, message=""):
        super(EmptyClusterError, self).__init__()
        self.message = message

    def __str__(self):
        if len(self.message) > 0:
            suffix = " (%s)" % self.message
        else:
            suffix = ""
        return "Cluster assignments lead to at least one empty cluster" + suffix


def _check_no_empty_cluster(labels, n_clusters):
    """Check that all clusters have at least one sample assigned.

    Examples
    --------
    >>> labels = numpy.array([1, 1, 2, 0, 2])
    >>> _check_no_empty_cluster(labels, 3)
    >>> _check_no_empty_cluster(labels, 4)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    EmptyClusterError: Cluster assignments lead to at least one empty cluster
    """

    for k in range(n_clusters):
        if numpy.sum(labels == k) == 0:
            raise EmptyClusterError


def _compute_inertia(distances, assignments, squared=True):
    """Derive inertia (average of squared distances) from pre-computed distances and assignments.

    Examples
    --------
    >>> dists = numpy.array([[1., 2., 0.5], [0., 3., 1.]])
    >>> assign = numpy.array([2, 0])
    >>> _compute_inertia(dists, assign)
    0.125
    """
    n_ts = distances.shape[0]
    if squared:
        return numpy.sum(distances[numpy.arange(n_ts), assignments] ** 2) / n_ts
    else:
        return numpy.sum(distances[numpy.arange(n_ts), assignments]) / n_ts


def silhouette_score(X, labels, metric=None, sample_size=None, metric_params=None,
                     random_state=None, **kwds):
    """Compute the mean Silhouette Coefficient of all samples (cf.  [1]_ and  [2]_).

    Read more in the `scikit-learn documentation
    <http://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient>`_.

    Parameters
    ----------
    X : array [n_ts, n_ts] if metric == "precomputed", or, \
             [n_ts, sz, d] otherwise
        Array of pairwise distances between time series, or a time series dataset.
    labels : array, shape = [n_ts]
         Predicted labels for each time series.
    metric : string, or callable
        The metric to use when calculating distance between time series.
        Should be one of {'dtw', 'softdtw', 'euclidean'} or a callable distance
        function.
        If X is the distance array itself, use ``metric="precomputed"``.
    sample_size : int or None
        The size of the sample to use when computing the Silhouette Coefficient
        on a random subset of the data.
        If ``sample_size is None``, no sampling is used.
    metric_params : dict or None
        Parameter values for the chosen metric. Value associated to the `"gamma_sdtw"` key corresponds to the gamma
        parameter in Soft-DTW.
    random_state : int, RandomState instance or None, optional (default=None)
        The generator used to randomly select a subset of samples.  If int,
        random_state is the seed used by the random number generator; If
        RandomState instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance used by
        `np.random`. Used when ``sample_size is not None``.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
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
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> labels = numpy.random.randint(2, size=50)
    >>> s_sc = silhouette_score(X, labels, metric="dtw")
    >>> s_sc2 = silhouette_score(X, labels, metric="softdtw")
    >>> s_sc2b = silhouette_score(X, labels, metric="softdtw", metric_params={"gamma_sdtw": 2.})
    >>> s_sc3 = silhouette_score(cdist_dtw(X), labels, metric="precomputed")
    """
    sklearn_metric = None
    if metric_params is None:
        metric_params = {}
    if metric == "precomputed":
        sklearn_X = X
    elif metric == "dtw":
        sklearn_X = cdist_dtw(X)
    elif metric == "softdtw":
        gamma = metric_params.get("gamma_sdtw", None)
        if gamma is not None:
            sklearn_X = cdist_soft_dtw(X, gamma=gamma)
        else:
            sklearn_X = cdist_soft_dtw(X)
    elif metric == "euclidean":
        sklearn_X = cdist(X, X, metric="euclidean")
    else:
        X_ = to_time_series_dataset(X)
        n, sz, d = X_.shape
        sklearn_X = X_.reshape((n, -1))
        if metric is None:
            metric = dtw
        sklearn_metric = lambda x, y: metric(to_time_series(x.reshape((sz, d)), remove_nans=True),
                                             to_time_series(y.reshape((sz, d)), remove_nans=True))
    return sklearn_silhouette_score(X=sklearn_X,
                                    labels=labels,
                                    metric="precomputed" if sklearn_metric is None else sklearn_metric,
                                    sample_size=sample_size,
                                    random_state=random_state,
                                    **kwds)


class GlobalAlignmentKernelKMeans(BaseEstimator, ClusterMixin):
    """Global Alignment Kernel K-means.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.
    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm stops.
    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.
    sigma : float (default: 1.)
        Bandwidth parameter for the Global Alignment kernel
    verbose : bool (default: True)
        Whether or not to print information about the inertia while learning the model.
    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it fixes the seed. Defaults to the global
        numpy random number generator.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center (computed using the kernel trick).

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> gak_km = GlobalAlignmentKernelKMeans(n_clusters=3, verbose=False, random_state=0).fit(X)
    >>> numpy.alltrue(gak_km.labels_ == gak_km.predict(X))
    True
    >>> numpy.alltrue(gak_km.fit(X).predict(X) == gak_km.fit_predict(X))
    True
    >>> GlobalAlignmentKernelKMeans(n_clusters=101, verbose=False, random_state=0).fit(X).X_fit_ is None
    True

    References
    ----------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.

    Fast Global Alignment Kernels.
    Marco Cuturi.
    ICML 2011.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-6, n_init=1, sigma=1., verbose=True, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.sigma = sigma
        self.n_init = n_init
        self.verbose = verbose
        self.max_attempts = max(self.n_init, 10)

        self.labels_ = None
        self.inertia_ = None
        self.sample_weight_ = None
        self.X_fit_ = None

    def _get_kernel(self, X, Y=None):
        return cdist_gak(X, Y, sigma=self.sigma)

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

        return self

    def fit(self, X, y=None, sample_weight=None):
        """Compute kernel k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        sample_weight : array-like of shape=(n_ts, ) or None (default: None)
            Weights to be given to time series in the learning process. By default, all time series weights are equal.
        """

        n_samples = X.shape[0]
        K = self._get_kernel(X)
        sw = sample_weight if sample_weight else numpy.ones(n_samples)
        self.sample_weight_ = sw
        rs = check_random_state(self.random_state)

        last_correct_labels = None
        min_inertia = numpy.inf
        n_attempts = 0
        n_successful = 0
        while n_successful < self.n_init and n_attempts < self.max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(K, rs)
                if self.inertia_ < min_inertia:
                    last_correct_labels = self.labels_
                    min_inertia = self.inertia_
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        if n_successful > 0:
            self.X_fit_ = X
            self.labels_ = last_correct_labels
            self.inertia_ = min_inertia
        else:
            self.X_fit_ = None
        return self

    def _compute_dist(self, K, dist):
        """Compute a n_samples x n_clusters distance matrix using the kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = (self.labels_ == j)

            if numpy.sum(mask) == 0:
                raise EmptyClusterError("try smaller n_cluster or better kernel parameters")

            # NB: we use a normalized kernel so k(x,x) = 1 for all x (including the centroid)
            dist[:, j] = 2 - 2 * numpy.sum(sw[mask] * K[:, mask], axis=1) / sw[mask].sum()

    @staticmethod
    def _compute_inertia(dist_sq):
        return dist_sq.min(axis=1).sum()

    def fit_predict(self, X, y=None):
        """Fit kernel k-means clustering using X and then predict the closest cluster each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

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
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = numpy.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist)
        return dist.argmin(axis=1)


class TimeSeriesCentroidBasedClusteringMixin:
    """Mixin class for centroid-based clustering of time series."""
    def _post_fit(self, X_fitted, centroids, inertia):
        if numpy.isfinite(inertia) and (centroids is not None):
            self.cluster_centers_ = centroids
            self._assign(X_fitted)
            self.X_fit_ = X_fitted
            self.inertia_ = inertia
        else:
            self.X_fit_ = None


class TimeSeriesKMeans(BaseEstimator, ClusterMixin, TimeSeriesCentroidBasedClusteringMixin):
    """K-means clustering for time-series data.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.
    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm stops.
    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.
    metric : {"euclidean", "dtw", "softdtw"} (default: "euclidean")
        Metric to be used for both cluster assignment and barycenter computation. If "dtw", DBA is used for barycenter
        computation.
    max_iter_barycenter : int (default: 100)
        Number of iterations for the barycenter computation process. Only used if `metric="dtw"` or `metric="softdtw"`.
    metric_params : dict or None
        Parameter values for the chosen metric. Value associated to the `"gamma_sdtw"` key corresponds to the gamma
        parameter in Soft-DTW.
    dtw_inertia: bool
        Whether to compute DTW inertia even if DTW is not the chosen metric.
    verbose : bool (default: True)
        Whether or not to print information about the inertia while learning the model.
    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it fixes the seed. Defaults to the global
        numpy random number generator.

    Attributes
    ----------
    labels_ : numpy.ndarray
        Labels of each point.
    cluster_centers_ : numpy.ndarray
        Cluster centers.
    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Note
    ----
        If `metric` is set to `"euclidean"`, the algorithm expects a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5, verbose=False, random_state=0).fit(X)
    >>> km.cluster_centers_.shape
    (3, 32, 1)
    >>> dists = cdist(X.reshape((50, 32)), km.cluster_centers_.reshape((3, 32)))
    >>> numpy.alltrue(km.labels_ == dists.argmin(axis=1))
    True
    >>> numpy.alltrue(km.labels_ == km.predict(X))
    True
    >>> numpy.alltrue(km.fit(X).predict(X) == km.fit_predict(X))
    True
    >>> km_dba = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5, max_iter_barycenter=5, verbose=False, \
                                  random_state=0).fit(X)
    >>> km_dba.cluster_centers_.shape
    (3, 32, 1)
    >>> dists = cdist_dtw(X, km_dba.cluster_centers_)
    >>> numpy.alltrue(km_dba.labels_ == dists.argmin(axis=1))
    True
    >>> numpy.alltrue(km_dba.labels_ == km_dba.predict(X))
    True
    >>> numpy.alltrue(km_dba.fit(X).predict(X) == km_dba.fit_predict(X))
    True
    >>> km_sdtw = TimeSeriesKMeans(n_clusters=3, metric="softdtw", max_iter=5, max_iter_barycenter=5, \
                                   metric_params={"gamma_sdtw": .5}, verbose=False, random_state=0).fit(X)
    >>> km_sdtw.cluster_centers_.shape
    (3, 32, 1)
    >>> dists = cdist_soft_dtw(X, km_sdtw.cluster_centers_, gamma=.5)
    >>> numpy.alltrue(km_sdtw.labels_ == dists.argmin(axis=1))
    True
    >>> numpy.alltrue(km_sdtw.labels_ == km_sdtw.predict(X))
    True
    >>> numpy.alltrue(km_sdtw.fit(X).predict(X) == km_sdtw.fit_predict(X))
    True
    >>> TimeSeriesKMeans(n_clusters=101, verbose=False, random_state=0).fit(X).X_fit_ is None
    True
    >>> X_bis = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3], [2, 5, 6, 7, 8, 9]])
    >>> km = TimeSeriesKMeans(n_clusters=2, verbose=False, max_iter=5, metric="dtw", random_state=0).fit(X_bis)
    >>> km_bis = TimeSeriesKMeans(n_clusters=2, verbose=False, max_iter=5, metric="softdtw", random_state=0).fit(X_bis)
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-6, n_init=1, metric="euclidean", max_iter_barycenter=100,
                 metric_params=None, dtw_inertia=False, verbose=True, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.metric = metric
        self.n_init = n_init
        self.verbose = verbose
        self.max_iter_barycenter = max_iter_barycenter
        self.max_attempts = max(self.n_init, 10)
        self.dtw_inertia = dtw_inertia

        self.labels_ = None
        self.inertia_ = numpy.inf
        self.cluster_centers_ = None
        self.X_fit_ = None
        self._squared_inertia = True

        if metric_params is None:
            metric_params = {}
        self.gamma_sdtw = metric_params.get("gamma_sdtw", 1.)

    def _fit_one_init(self, X, x_squared_norms, rs):
        n_ts, _, d = X.shape
        sz = min([ts_size(ts) for ts in X])
        self.cluster_centers_ = _k_init(X[:, :sz, :].reshape((n_ts, -1)),
                                        self.n_clusters, x_squared_norms, rs).reshape((-1, sz, d))
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

        return self

    def _assign(self, X, update_class_attributes=True):
        if self.metric == "euclidean":
            dists = cdist(X.reshape((X.shape[0], -1)), self.cluster_centers_.reshape((self.n_clusters, -1)),
                          metric="euclidean")
        elif self.metric == "dtw":
            dists = cdist_dtw(X, self.cluster_centers_)
        elif self.metric == "softdtw":
            dists = cdist_soft_dtw(X, self.cluster_centers_, gamma=self.gamma_sdtw)
        else:
            raise ValueError("Incorrect metric: %s (should be one of 'dtw', 'softdtw', 'euclidean')" % self.metric)
        matched_labels = dists.argmin(axis=1)
        if update_class_attributes:
            self.labels_ = matched_labels
            _check_no_empty_cluster(self.labels_, self.n_clusters)
            if self.dtw_inertia and self.metric != "dtw":
                inertia_dists = cdist_dtw(X, self.cluster_centers_)
            else:
                inertia_dists = dists
            self.inertia_ = _compute_inertia(inertia_dists, self.labels_, self._squared_inertia)
        return matched_labels

    def _update_centroids(self, X):
        for k in range(self.n_clusters):
            if self.metric == "dtw":
                self.cluster_centers_[k] = DTWBarycenterAveraging(max_iter=self.max_iter_barycenter,
                                                                  barycenter_size=None,
                                                                  init_barycenter=self.cluster_centers_[k],
                                                                  verbose=False).fit(X[self.labels_ == k])
            elif self.metric == "softdtw":
                self.cluster_centers_[k] = SoftDTWBarycenter(max_iter=self.max_iter_barycenter,
                                                             gamma=self.gamma_sdtw,
                                                             init=self.cluster_centers_[k]).fit(X[self.labels_ == k])
            else:
                self.cluster_centers_[k] = EuclideanBarycenter().fit(X[self.labels_ == k])

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        """
        n_successful = 0

        X_ = to_time_series_dataset(X)
        rs = check_random_state(self.random_state)
        x_squared_norms = cdist(X_.reshape((X_.shape[0], -1)), numpy.zeros((1, X_.shape[1] * X_.shape[2])),
                                metric="sqeuclidean").reshape((1, -1))

        best_correct_centroids = None
        min_inertia = numpy.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < self.max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(X_, x_squared_norms, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                n_successful += 1
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self

    def fit_predict(self, X, y=None):
        """Fit k-means clustering using X and then predict the closest cluster each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

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
        X_ = to_time_series_dataset(X)
        return self._assign(X_, update_class_attributes=False)


class KShape(BaseEstimator, ClusterMixin, TimeSeriesCentroidBasedClusteringMixin):
    """KShape clustering for time series.

    KShape was originally presented in [1]_.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.
    max_iter : int (default: 100)
        Maximum number of iterations of the k-Shape algorithm.
    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm stops.
    n_init : int (default: 1)
        Number of time the k-Shape algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.
    verbose : bool (default: True)
        Whether or not to print information about the inertia while learning the model.
    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it fixes the seed. Defaults to the global
        numpy random number generator.

    Attributes
    ----------
    cluster_centers_ : numpy.ndarray of shape (sz, d).
        Centroids
    labels_ : numpy.ndarray of integers with shape (n_ts, ).
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Note
    ----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> X = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X)
    >>> ks = KShape(n_clusters=3, n_init=1, verbose=False, random_state=0).fit(X)
    >>> ks.cluster_centers_.shape
    (3, 32, 1)
    >>> dists = ks._cross_dists(X)
    >>> numpy.alltrue(ks.labels_ == dists.argmin(axis=1))
    True
    >>> numpy.alltrue(ks.labels_ == ks.predict(X))
    True
    >>> numpy.alltrue(ks.predict(X) == KShape(n_clusters=3, n_init=1, verbose=False, random_state=0).fit_predict(X))
    True
    >>> KShape(n_clusters=101, verbose=False, random_state=0).fit(X).X_fit_ is None
    True

    References
    ----------
    .. [1] J. Paparrizos & L. Gravano. k-Shape: Efficient and Accurate Clustering of Time Series. SIGMOD 2015.
       pp. 1855-1870.
    """
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-6, n_init=1, verbose=True, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.verbose = verbose
        self.max_attempts = max(self.n_init, 10)

        self.labels_ = None
        self.inertia_ = numpy.inf
        self.cluster_centers_ = None

        self._norms = 0.
        self._norms_centroids = 0.

    def _shape_extraction(self, X, k):
        sz = X.shape[1]
        Xp = y_shifted_sbd_vec(self.cluster_centers_[k], X[self.labels_ == k], norm_ref=-1,
                               norms_dataset=self._norms[self.labels_ == k])
        S = numpy.dot(Xp[:, :, 0].T, Xp[:, :, 0])
        Q = numpy.eye(sz) - numpy.ones((sz, sz)) / sz
        M = numpy.dot(Q.T, numpy.dot(S, Q))
        _, vec = numpy.linalg.eigh(M)
        mu_k = vec[:, -1].reshape((sz, 1))

        # The way the optimization problem is (ill-)formulated, both mu_k and -mu_k are candidates for barycenters
        # In the following, we check which one is best candidate
        dist_plus_mu = numpy.sum(numpy.linalg.norm(Xp - mu_k, axis=(1, 2)))
        dist_minus_mu = numpy.sum(numpy.linalg.norm(Xp + mu_k, axis=(1, 2)))
        if dist_minus_mu < dist_plus_mu:
            mu_k *= -1

        return mu_k

    def _update_centroids(self, X):
        for k in range(self.n_clusters):
            self.cluster_centers_[k] = self._shape_extraction(X, k)
        self.cluster_centers_ = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(self.cluster_centers_)
        self._norms_centroids = numpy.linalg.norm(self.cluster_centers_, axis=(1, 2))

    def _cross_dists(self, X):
        return 1. - cdist_normalized_cc(X, self.cluster_centers_, norms1=self._norms,
                                        norms2=self._norms_centroids, self_similarity=False)

    def _assign(self, X):
        dists = self._cross_dists(X)
        self.labels_ = dists.argmin(axis=1)
        _check_no_empty_cluster(self.labels_, self.n_clusters)
        self.inertia_ = _compute_inertia(dists, self.labels_)

    def _fit_one_init(self, X, rs):
        n_samples, sz, d = X.shape
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)
        self.cluster_centers_ = rs.randn(self.n_clusters, sz, d)
        self._norms_centroids = numpy.linalg.norm(self.cluster_centers_, axis=(1, 2))
        old_inertia = numpy.inf

        for it in range(self.max_iter):
            old_cluster_centers = self.cluster_centers_.copy()
            self._update_centroids(X)
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")

            if numpy.abs(old_inertia - self.inertia_) < self.tol:
                break
            if old_inertia - self.inertia_ < 0:
                self.cluster_centers_ = old_cluster_centers
                self._assign(X)
                break

            old_inertia = self.inertia_
        if self.verbose:
            print("")

        return self

    def fit(self, X, y=None):
        """Compute k-Shape clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        """

        X_ = to_time_series_dataset(X)
        self._norms = numpy.linalg.norm(X_, axis=(1, 2))
        X_ = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_)
        assert X_.shape[-1] == 1, "kShape is supposed to work on monomodal data, provided data has dimension %d" % \
                                  X_.shape[-1]
        rs = check_random_state(self.random_state)

        best_correct_centroids = None
        min_inertia = numpy.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < self.max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(X_, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self._post_fit(X_, best_correct_centroids, min_inertia)
        self._norms_centroids = numpy.linalg.norm(self.cluster_centers_, axis=(1, 2))
        return self

    def fit_predict(self, X, y=None):
        """Fit k-Shape clustering using X and then predict the closest cluster each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_

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
        X_ = to_time_series_dataset(X)
        X_ = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_)
        dists = self._cross_dists(X_)
        return dists.argmin(axis=1)
