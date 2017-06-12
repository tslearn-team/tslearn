import numpy
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

from tslearn.metrics import cdist_gak, cdist_dtw
from tslearn.barycenters import EuclideanBarycenter, DTWBarycenterAveraging
from tslearn.utils import npy3d_time_series_dataset


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
# Kernel k-means is derived from https://gist.github.com/mblondel/6230787 by Mathieu Blondel
# License: BSD 3 clause

class EmptyClusterError(Exception):
    def __init__(self, message=""):
        super(EmptyClusterError, self).__init__(message)


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
            for k in range(self.n_clusters):
                if numpy.sum(self.labels_ == k) == 0:
                    raise EmptyClusterError
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
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset.
        sample_weight : array-like, shape=(n_ts, ), optional
            Weights to be given to time series in the learning process. By default, all time series weights are equal.
        """
        n_successful = 0

        n_samples = X.shape[0]
        K = self._get_kernel(X)
        sw = sample_weight if sample_weight else numpy.ones(n_samples)
        self.sample_weight_ = sw
        rs = check_random_state(self.random_state)

        last_correct_labels = None
        min_inertia = numpy.inf
        for trial in range(self.n_init):
            try:
                if self.verbose:
                    print("Init %d" % (trial + 1))
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
            mask = self.labels_ == j

            if numpy.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster or better kernel parameters.")

            # NB: we use a normalized kernel so k(x,x) = 1 for all x (including the centroid)
            dist[:, j] = 2 - 2 * numpy.sum(sw[mask] * K[:, mask], axis=1) / sw[mask].sum()

    @staticmethod
    def _compute_inertia(dist):
        return dist.min(axis=1).sum()

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array, shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = numpy.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist)
        return dist.argmin(axis=1)


class TimeSeriesKMeansOld(KMeans):
    """Standard Euclidean K-Means clustering for time series data.

    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    algorithm : "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.

    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances

        False : never precompute distances

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.

    verbose : int, default 0
        Verbosity mode.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    Attributes
    ----------
    cluster_centers_ : array of shape (n_clusters, sz, d)
        Coordinates of cluster centers

    labels_ :
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.
    """
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto'):
        KMeans.__init__(self, n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                        precompute_distances=precompute_distances, verbose=verbose, random_state=random_state,
                        copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm)
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Training time series dataset to cluster.
        """
        X = self._check_fit_data(X)
        n_ts, sz, d = X.shape
        X_ = X.reshape((n_ts, -1))
        # TODO: The following is ugly, I just do not know why using inheritance does not work (KMeans.fit(self, X_, y))
        km = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, max_iter=self.max_iter,
                    tol=self.tol, precompute_distances=self.precompute_distances, verbose=self.verbose,
                    random_state=self.random_state, copy_x=self.copy_x, n_jobs=self.n_jobs, algorithm=self.algorithm)
        km.fit(X_, y)
        self.cluster_centers_ = km.cluster_centers_.reshape((-1, sz, d))
        self.labels_ = km.labels_
        self.inertia_ = km.inertia_
        return self

    def _check_fit_data(self, X):
        """Verify that the number of samples given is larger than k"""
        X = npy3d_time_series_dataset(X)
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (X.shape[0], self.n_clusters))
        return X

    def _check_test_data(self, X):
        X = npy3d_time_series_dataset(X)
        n_ts, sz, d = X.shape
        expected_sz, expected_d = self.cluster_centers_.shape[1:]
        if not (sz == expected_sz and d == expected_d):
            raise ValueError("Incorrect shape. Got size %d and dimension %d, expected size %d and dimension %d" %
                             (sz, d, expected_sz, expected_d))

        return X

    def transform(self, X, y=None):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset to transform.
        Returns
        -------
        X_new : array, shape=(n_samples, k)
            X transformed in the new space.
        """
        X = self._check_fit_data(X)
        return self._transform(X, y)

    def _transform(self, X, y=None):
        n_ts, sz, d = X.shape
        return euclidean_distances(X.reshape((n_ts, -1)), self.cluster_centers_.reshape((self.n_clusters, -1)))

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset to predict.
        Returns
        -------
        labels : array, shape=(n_ts,)
            Index of the cluster each time series belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        n_ts, sz, d = X.shape
        dists = euclidean_distances(X.reshape((n_ts, -1)), self.cluster_centers_.reshape((self.n_clusters, -1)))
        return numpy.argmin(dists, axis=1)


class TimeSeriesKMeans(BaseEstimator, ClusterMixin):
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
    metric : {"euclidean", "dtw"}, default: "euclidean"
        Metric to be used for both cluster assignment and barycenter computation. If "dtw", DBA is used for barycenter
        computation.
    n_iter_dba : int (default: 100)
        Number of iterations for the DBA barycenter computation process. Only used if `metric="dtw"`.
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
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-6, n_init=1, metric="euclidean", n_iter_dba=100, verbose=True,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.metric = metric
        self.n_init = n_init
        self.verbose = verbose

        self.labels_ = None
        self.inertia_ = numpy.inf
        self.cluster_centers_ = None
        self.X_fit_ = None

        if self.metric == "dtw":
            self.dba_ = DTWBarycenterAveraging(n_iter=n_iter_dba, barycenter_size=None, verbose=False)

    def _fit_one_init(self, X, x_squared_norms, rs):
        n_samples, sz, d = X.shape
        self.cluster_centers_ = _k_init(X.reshape((n_samples, -1)),
                                        self.n_clusters, x_squared_norms, rs).reshape((-1, sz, d))
        old_inertia = numpy.inf

        for it in range(self.max_iter):
            self._assign(X)
            self._update_centroids(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")

            if numpy.abs(old_inertia - self.inertia_) < self.tol:
                break
            old_inertia = self.inertia_
        if self.verbose:
            print("")

        return self

    def _assign(self, X):
        if self.metric == "euclidean":
            dists = cdist(X.reshape((X.shape[0], -1)), self.cluster_centers_.reshape((self.n_clusters, -1)),
                          metric="euclidean")
        elif self.metric == "dtw":
            dists = cdist_dtw(X, self.cluster_centers_)
        else:
            raise ValueError("Incorrect metric: %s (should be one of 'dtw', 'euclidean')" % self.metric)
        self.labels_ = dists.argmin(axis=1)
        for k in range(self.n_clusters):
            if numpy.sum(self.labels_ == k) == 0:
                raise EmptyClusterError
        self.inertia_ = numpy.sum(dists[numpy.arange(X.shape[0]), self.labels_] ** 2) / X.shape[0]

    def _update_centroids(self, X):
        for k in range(self.n_clusters):
            if self.metric == "euclidean":
                self.cluster_centers_[k] = EuclideanBarycenter().fit(X[self.labels_ == k])
            elif self.metric == "dtw":
                self.cluster_centers_[k] = self.dba_.fit(X[self.labels_ == k])
            else:
                raise ValueError("Incorrect metric: %s (should be one of 'dtw', 'euclidean')" % self.metric)

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset.
        """
        n_successful = 0

        X_ = npy3d_time_series_dataset(X)
        rs = check_random_state(self.random_state)
        x_squared_norms = cdist(X_.reshape((X_.shape[0], -1)), numpy.zeros((1, X_.shape[1] * X_.shape[2])),
                                metric="sqeuclidean").reshape((1, -1))

        last_correct_centroids = None
        min_inertia = numpy.inf
        for trial in range(self.n_init):
            try:
                if self.verbose:
                    print("Init %d" % (trial + 1))
                self._fit_one_init(X_, x_squared_norms, rs)
                if self.inertia_ < min_inertia:
                    last_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        if n_successful > 0:
            self.X_fit_ = X_
            self.cluster_centers_ = last_correct_centroids
            self._assign(X_)
            self.inertia_ = min_inertia
        else:
            self.X_fit_ = None
        return self

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array, shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = numpy.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist)
        return dist.argmin(axis=1)