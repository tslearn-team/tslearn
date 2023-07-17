# Code for soft DTW is by Mathieu Blondel under Simplified BSD license

import numpy
from scipy.optimize import minimize

from tslearn.utils import to_time_series_dataset, check_equal_size, \
    to_time_series
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.metrics import SquaredEuclidean, SoftDTW

from .utils import _set_weights
from .euclidean import euclidean_barycenter

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _softdtw_func(Z, X, weights, barycenter, gamma):
    # Compute objective value and grad at Z.

    Z = Z.reshape(barycenter.shape)
    G = numpy.zeros_like(Z)
    obj = 0

    for i in range(len(X)):
        D = SquaredEuclidean(Z, X[i])
        sdtw = SoftDTW(D, gamma=gamma)
        value = sdtw.compute()
        E = sdtw.grad()
        G_tmp = D.jacobian_product(E)
        G += weights[i] * G_tmp
        obj += weights[i] * value

    return obj, G.ravel()


def softdtw_barycenter(X, gamma=1.0, weights=None, method="L-BFGS-B", tol=1e-3,
                       max_iter=50, init=None):
    """Compute barycenter (time series averaging) under the soft-DTW [1]
    geometry.

    Soft-DTW was originally presented in [1]_.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.
    gamma: float
        Regularization parameter.
        Lower is less smoothed (closer to true DTW).
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.
    method: string
        Optimization method, passed to `scipy.optimize.minimize`.
        Default: L-BFGS.
    tol: float
        Tolerance of the method used.
    max_iter: int
        Maximum number of iterations.
    init: array or None (default: None)
        Initial barycenter to start from for the optimization process.
        If `None`, euclidean barycenter is used as a starting point.

    Returns
    -------
    numpy.array of shape (bsz, d) where `bsz` is the size of the `init` array \
            if provided or `sz` otherwise
        Soft-DTW barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> softdtw_barycenter(time_series, max_iter=5)
    array([[1.25161574],
           [2.03821705],
           [3.5101956 ],
           [4.36140605]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> softdtw_barycenter(time_series, max_iter=5)
    array([[1.21349933],
           [1.8932251 ],
           [2.67573269],
           [3.51057026],
           [4.33645802]])

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    X_ = to_time_series_dataset(X)
    weights = _set_weights(weights, X_.shape[0])
    if init is None:
        if check_equal_size(X_):
            barycenter = euclidean_barycenter(X_, weights)
        else:
            resampled_X = TimeSeriesResampler(sz=X_.shape[1]).fit_transform(X_)
            barycenter = euclidean_barycenter(resampled_X, weights)
    else:
        barycenter = init

    if max_iter > 0:
        X_ = [to_time_series(d, remove_nans=True) for d in X_]

        def f(Z):
            return _softdtw_func(Z, X_, weights, barycenter, gamma)

        # The function works with vectors so we need to vectorize barycenter.
        res = minimize(f, barycenter.ravel(), method=method, jac=True, tol=tol,
                       options=dict(maxiter=max_iter, disp=False))
        return res.x.reshape(barycenter.shape)
    else:
        return barycenter
