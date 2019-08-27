"""
The :mod:`tslearn.barycenters` module gathers algorithms for time series
barycenter computation.
"""

# Code for soft DTW is by Mathieu Blondel under Simplified BSD license

import numpy
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
import warnings

from tslearn.utils import to_time_series_dataset, check_equal_size, \
    to_time_series
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.metrics import dtw_path, SquaredEuclidean, SoftDTW


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _set_weights(w, n):
    """Return w if it is a valid weight vector of size n, and a vector of n 1s
    otherwise.
    """
    if w is None or len(w) != n:
        w = numpy.ones((n, ))
    return w


def euclidean_barycenter(X, weights=None):
    """Standard Euclidean barycenter computed from a set of time series.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    Returns
    -------
    numpy.array of shape (sz, d)
        Barycenter of the provided time series dataset.

    Notes
    -----
        This method requires a dataset of equal-sized time series

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> bar = euclidean_barycenter(time_series)
    >>> bar.shape
    (4, 1)
    >>> bar
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    """
    X_ = to_time_series_dataset(X)
    weights = _set_weights(weights, X_.shape[0])
    return numpy.average(X_, axis=0, weights=weights)


def _init_avg(X, barycenter_size):
    if X.shape[1] == barycenter_size:
        return numpy.nanmean(X, axis=0)
    else:
        X_avg = numpy.nanmean(X, axis=0)
        xnew = numpy.linspace(0, 1, barycenter_size)
        f = interp1d(numpy.linspace(0, 1, X_avg.shape[0]), X_avg,
                     kind="linear", axis=0)
        return f(xnew)


def _petitjean_assignment(X, barycenter):
    n = X.shape[0]
    barycenter_size = barycenter.shape[0]
    assign = ([[] for _ in range(barycenter_size)],
              [[] for _ in range(barycenter_size)])
    for i in range(n):
        path, _ = dtw_path(X[i], barycenter)
        for pair in path:
            assign[0][pair[1]].append(i)
            assign[1][pair[1]].append(pair[0])
    return assign


def _petitjean_update_barycenter(X, assign, barycenter_size, weights):
    barycenter = numpy.zeros((barycenter_size, X.shape[-1]))
    for t in range(barycenter_size):
        barycenter[t] = numpy.average(X[assign[0][t], assign[1][t]], axis=0,
                                      weights=weights[assign[0][t]])
    return barycenter


def _petitjean_cost(X, barycenter, assign, weights):
    cost = 0.
    barycenter_size = barycenter.shape[0]
    for t_barycenter in range(barycenter_size):
        for i_ts, t_ts in zip(assign[0][t_barycenter],
                              assign[1][t_barycenter]):
            sq_norm = numpy.linalg.norm(X[i_ts, t_ts] -
                                        barycenter[t_barycenter]) ** 2
            cost += weights[i_ts] * sq_norm
    return cost / weights.sum()


def dtw_barycenter_averaging(X, barycenter_size=None, init_barycenter=None,
                             max_iter=30, tol=1e-5, weights=None,
                             verbose=False):
    """DTW Barycenter Averaging (DBA) method.

    DBA was originally presented in [1]_.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.
    barycenter_size : int or None (default: None)
        Size of the barycenter to generate. If None, the size of the barycenter
        is that of the data provided at fit
        time or that of the initial barycenter if specified.
    init_barycenter : array or None (default: None)
        Initial barycenter to start from for the optimization process.
    max_iter : int (default: 30)
        Number of iterations of the Expectation-Maximization optimization
        procedure.
    tol : float (default: 1e-5)
        Tolerance to use for early stopping: if the decrease in cost is lower
        than this value, the
        Expectation-Maximization procedure stops.
    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.
    verbose : boolean (default: False)
        Whether to print information about the cost at each iteration or not.

    Returns
    -------
    numpy.array of shape (barycenter_size, d) or (sz, d) if barycenter_size \
            is None
        DBA barycenter of the provided time series dataset.

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> dtw_barycenter_averaging(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    >>> time_series = [[1, 2, 3, 4], [1, 2, 3, 4, 5]]
    >>> dtw_barycenter_averaging(time_series, max_iter=5)
    array([[1. ],
           [2. ],
           [3. ],
           [4. ],
           [4.5]])
    >>> dtw_barycenter_averaging(time_series, max_iter=5, barycenter_size=3)
    array([[1.5       ],
           [3.        ],
           [4.33333333]])

    References
    ----------
    .. [1] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    """
    X_ = to_time_series_dataset(X)
    if barycenter_size is None:
        barycenter_size = X_.shape[1]
    weights = _set_weights(weights, X_.shape[0])
    if init_barycenter is None:
        barycenter = _init_avg(X_, barycenter_size)
    else:
        barycenter_size = init_barycenter.shape[0]
        barycenter = init_barycenter
    cost_prev, cost = numpy.inf, numpy.inf
    for it in range(max_iter):
        assign = _petitjean_assignment(X_, barycenter)
        cost = _petitjean_cost(X_, barycenter, assign, weights)
        if verbose:
            print("[DBA] epoch %d, cost: %.3f" % (it + 1, cost))
        barycenter = _petitjean_update_barycenter(X_, assign, barycenter_size,
                                                  weights)
        if abs(cost_prev - cost) < tol:
            break
        elif cost_prev < cost:
            warnings.warn("DBA loss is increasing while it should not be. "
                          "Stopping optimization.", ConvergenceWarning)
            break
        else:
            cost_prev = cost
    return barycenter


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
    """Compute barycenter (time series averaging) under the soft-DTW geometry.

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
        X_ = numpy.array([to_time_series(d, remove_nans=True) for d in X_])

        def f(Z):
            return _softdtw_func(Z, X_, weights, barycenter, gamma)

        # The function works with vectors so we need to vectorize barycenter.
        res = minimize(f, barycenter.ravel(), method=method, jac=True, tol=tol,
                       options=dict(maxiter=max_iter, disp=False))
        return res.x.reshape(barycenter.shape)
    else:
        return barycenter
