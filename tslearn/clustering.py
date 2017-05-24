import numpy
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from tslearn.metrics import cdist_gak


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
# Derived from https://gist.github.com/mblondel/6230787 by Mathieu Blondel
# License: BSD 3 clause


class GlobalAlignmentKernelKMeans(BaseEstimator, ClusterMixin):
    """Global Alignment Kernel K-means.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.
    max_iter : int (default: 50)
        Maximum number of iterations of the k-means algorithm for a single run.
    n_init : int (default: 1)
        Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.
    sigma : float (default: 1.)
        Bandwidth parameter for the Global Alignment kernel
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

    def __init__(self, n_clusters=3, max_iter=50, n_init=1, sigma=1., random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.sigma = sigma
        self.n_init = n_init

        self.labels_ = None
        self.within_distances_ = None
        self.sample_weight_ = None
        self.X_fit_ = None

    def _get_kernel(self, X, Y=None):
        return cdist_gak(X, Y, sigma=self.sigma)

    def _fit_one_trial(self, K, rs):
        n_samples = K.shape[0]

        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = numpy.empty((n_samples, self.n_clusters))
        self.within_distances_ = numpy.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_, update_within=True)
            self.labels_ = dist.argmin(axis=1)

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
        for trial in range(self.n_init):
            try:
                self._fit_one_trial(K, rs)  # TODO: assess clustering quality
                last_correct_labels = self.labels_
                n_successful += 1
            except ValueError:
                pass
        if n_successful > 0:
            self.X_fit_ = X
            self.labels_ = last_correct_labels
        else:
            self.X_fit_ = None
        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if numpy.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster or better kernel parameters.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = numpy.sum(numpy.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
            dist[:, j] += within_distances[j] - 2 * numpy.sum(sw[mask] * K[:, mask], axis=1) / denom

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
        self._compute_dist(K, dist, self.within_distances_, update_within=False)
        return dist.argmin(axis=1)

