import numpy
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from tslearn.metrics import cdist_gak


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
# Adapted from https://gist.github.com/mblondel/6230787 by Mathieu Blondel


class GlobalAlignmentKernelKMeans(BaseEstimator, ClusterMixin):
    """Global Alignment Kernel K-means

    References
    ----------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.

    Fast Global Alignment Kernels.
    Marco Cuturi.
    ICML 2011.
    """

    def __init__(self, n_clusters=3, max_iter=50, random_state=None, sigma=0., verbose=0):  # TODO: add n_init option
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.sigma = sigma
        self.verbose = verbose

        self.labels_ = None
        self.within_distances_ = None
        self.sample_weight_ = None
        self.X_fit_ = None

    def _get_kernel(self, X, Y=None):
        return cdist_gak(X, Y, sigma=self.sigma)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else numpy.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = numpy.zeros((n_samples, self.n_clusters))
        self.within_distances_ = numpy.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_, update_within=True)
            self.labels_ = dist.argmin(axis=1)

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if numpy.sum(mask) == 0:
                print(self.labels_ )
                raise ValueError("Empty cluster found, try smaller n_cluster or better kernel parameters.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = numpy.sum(numpy.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
            dist[:, j] += within_distances[j] - 2 * numpy.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = numpy.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_, update_within=False)
        return dist.argmin(axis=1)

