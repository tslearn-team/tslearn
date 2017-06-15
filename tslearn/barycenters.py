import numpy
from scipy.interpolate import interp1d

from tslearn.utils import npy3d_time_series_dataset
from tslearn.metrics import dtw_path


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class EuclideanBarycenter:
    """Standard Euclidean barycenter computed from a set of time series."""
    def fit(self, X):
        """Compute the barycenter from a dataset of time series.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        numpy.array of shape (sz, d)
            Barycenter of the provided time series dataset.
        """
        return npy3d_time_series_dataset(X).mean(axis=0)


class DTWBarycenterAveraging:
    """DBA barycenter as described in [1]_.

    Parameters
    ----------
    n_iter : int
        Number of iterations of the EM procedure.
    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter is that of the data provided at fit
        time.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower than this value, the EM procedure stops.
    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.
    
    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method for dynamic time warping, with
       applications to clustering. Pattern Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    def __init__(self, n_iter, barycenter_size=None, tol=1e-5, verbose=False):
        self.n_iter = n_iter
        self.barycenter_size = barycenter_size
        self.tol = tol
        self.verbose = verbose

    def fit(self, X):
        """Compute the barycenter from a dataset of time series.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size is None
            DBA barycenter of the provided time series dataset.
        """
        X_ = npy3d_time_series_dataset(X)
        if self.barycenter_size is None:
            self.barycenter_size = X_.shape[1]
        barycenter = self._init_avg(X_)
        cost_prev, cost = numpy.inf, numpy.inf
        for it in range(self.n_iter):
            assign = self._petitjean_assignment(X_, barycenter)
            barycenter = self._petitjean_update_barycenter(X_, assign)
            cost = self._petitjean_cost(X_, barycenter, assign)
            if self.verbose:
                print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
            if cost_prev - cost < self.tol:
                break
            else:
                cost_prev = cost
        return barycenter

    def _init_avg(self, X):
        if X.shape[1] == self.barycenter_size:
            return X.mean(axis=0)
        else:
            X_avg = X.mean(axis=0)
            xnew = numpy.linspace(0, 1, self.barycenter_size)
            f = interp1d(numpy.linspace(0, 1, X_avg.shape[0]), X_avg, kind="linear")
            return f(xnew)

    def _petitjean_assignment(self, X, barycenter):
        n = X.shape[0]
        assign = ([[] for _ in range(self.barycenter_size)], [[] for _ in range(self.barycenter_size)])
        for i in range(n):
            path, _ = dtw_path(X[i], barycenter)
            for pair in path:
                assign[0][pair[1]].append(i)
                assign[1][pair[1]].append(pair[0])
        return assign

    def _petitjean_update_barycenter(self, X, assign):
        barycenter = numpy.zeros((self.barycenter_size, X.shape[-1]))
        for t in range(self.barycenter_size):
            barycenter[t] = X[assign[0][t], assign[1][t]].mean(axis=0)
        return barycenter

    def _petitjean_cost(self, X, barycenter, assign):
        cost = 0.
        for t_barycenter in range(self.barycenter_size):
            for tt, (i_ts, t_ts) in enumerate(zip(assign[0][t_barycenter], assign[1][t_barycenter])):
                cost += numpy.linalg.norm(X[i_ts, t_ts] - barycenter[t_barycenter]) ** 2
        return cost / X.shape[0]
