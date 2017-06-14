import numpy
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster.k_means_ import _k_init
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist

from tslearn.metrics import cdist_gak, cdist_dtw
from tslearn.barycenters import EuclideanBarycenter, DTWBarycenterAveraging
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import npy3d_time_series_dataset
from tslearn.cycc import cdist_normalized_cc, y_shifted_sbd_vec


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
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        sample_weight : array-like of shape=(n_ts, ) or None (default: None)
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
    metric : {"euclidean", "dtw"} (default: "euclidean")
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

    def fit(self, X, y=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        """
        n_successful = 0

        X_ = npy3d_time_series_dataset(X)
        rs = check_random_state(self.random_state)
        x_squared_norms = cdist(X_.reshape((X_.shape[0], -1)), numpy.zeros((1, X_.shape[1] * X_.shape[2])),
                                metric="sqeuclidean").reshape((1, -1))

        best_correct_centroids = None
        min_inertia = numpy.inf
        for trial in range(self.n_init):
            try:
                if self.verbose:
                    print("Init %d" % (trial + 1))
                self._fit_one_init(X_, x_squared_norms, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        if n_successful > 0:
            self.X_fit_ = X_
            self.cluster_centers_ = best_correct_centroids
            self._assign(X_)
        else:
            self.X_fit_ = None
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
        X_ = npy3d_time_series_dataset(X)
        K = self._get_kernel(X_, self.X_fit_)
        n_samples = X_.shape[0]
        dist = numpy.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist)
        return dist.argmin(axis=1)


class KShape(BaseEstimator, ClusterMixin):
    """kShape clustering for time series as presented in [1]_.

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

        self.labels_ = None
        self.inertia_ = None
        self.cluster_centers_ = None

    def _shape_extraction(self, X, k):
        sz = X.shape[1]
        Xp = y_shifted_sbd_vec(self.cluster_centers_[k], X[self.labels_ == k], norm_ref=-1,
                               norms_dataset=numpy.array([-1.]))  # TODO: provide norms
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

    def _cross_dists(self, X):
        return 1. - cdist_normalized_cc(X, self.cluster_centers_, norms1=numpy.array([-1.]), norms2=numpy.array([-1.]),
                                        self_similarity=False)  # TODO: provide norms

    def _assign(self, X):
        dists = self._cross_dists(X)
        self.labels_ = dists.argmin(axis=1)
        for k in range(self.n_clusters):
            if numpy.sum(self.labels_ == k) == 0:
                raise EmptyClusterError
        self.inertia_ = numpy.sum(dists[numpy.arange(X.shape[0]), self.labels_] ** 2) / X.shape[0]

    def _fit_one_init(self, X, rs):
        n_samples, sz, d = X.shape
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)
        self.cluster_centers_ = numpy.random.randn(self.n_clusters, sz, d)
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
        n_successful = 0

        X_ = npy3d_time_series_dataset(X)
        X_ = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_)
        assert X_.shape[-1] == 1, "kShape is supposed to work on monomodal data, provided data has dimension %d" % \
                                  X_.shape[-1]
        rs = check_random_state(self.random_state)

        best_correct_centroids = None
        min_inertia = numpy.inf
        for trial in range(self.n_init):
            try:
                if self.verbose:
                    print("Init %d" % (trial + 1))
                self._fit_one_init(X_, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        if n_successful > 0:
            self.cluster_centers_ = best_correct_centroids
            self._assign(X_)
            self.inertia_ = min_inertia
        else:
            raise EmptyClusterError
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
        X_ = npy3d_time_series_dataset(X)
        X_ = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(X_)
        dists = self._cross_dists(X_)
        return dists.argmin(axis=1)


if __name__ == "__main__":
    import numpy
    import pylab

    from tslearn.generators import random_walk_blobs

    numpy.random.seed(0)
    X, y = random_walk_blobs(n_ts_per_blob=50, sz=128, d=1, n_blobs=3)
    ks = KShape(n_clusters=3, n_init=10, random_state=0)
    y_pred = ks.fit_predict(X)

    own_colors = ["r", "g", "b"]
    pylab.figure()
    for xx, yy in zip(X, y_pred):
        pylab.plot(numpy.arange(128), xx, own_colors[yy] + "-")
    pylab.show()


