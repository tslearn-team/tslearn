# Code for soft DTW is by Mathieu Blondel under Simplified BSD license

from joblib import Parallel, delayed

from numba import njit, prange

import numpy

from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from tslearn.backend import instantiate_backend
from tslearn.metrics._soft_dtw import _njit_accumulated_matrix_from_distance_matrix, _njit_soft_dtw_grad
from tslearn.metrics._masks import _njit_compute_mask, GLOBAL_CONSTRAINT_CODE
from tslearn.preprocessing import TimeSeriesResampler
from tslearn.utils import to_time_series_dataset
from tslearn.utils.utils import _check_equal_size, _to_time_series

from .utils import _set_weights

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _acc_softdtw_func(
    Z,
    X,
    weights,
    gamma,
    global_constraint=0,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None
):
    """Compute objective value and grad at Z for a given dataset."""

    Z = Z.reshape(-1, X[0].shape[-1])
    G = numpy.zeros_like(Z)
    obj = 0
    use_parallel = n_jobs not in [None, 1]

    if use_parallel:
        for obj_tmp, G_tmp in Parallel(n_jobs=n_jobs, prefer="threads", return_as="generator")(
            delayed(_softdtw_func)(
                Z,
                X[i],
                weights[i],
                gamma,
                global_constraint,
                sakoe_chiba_radius,
                itakura_max_slope
            ) for i in range(len(X))
        ):
            obj += obj_tmp
            G += G_tmp
    else:
        for i in range(len(X)):
            obj_tmp, G_tmp = _softdtw_func(
                Z,
                X[i],
                weights[i],
                gamma,
                global_constraint,
                sakoe_chiba_radius,
                itakura_max_slope
            )
            obj += obj_tmp
            G += G_tmp

    return obj, G.ravel()


def _softdtw_func(
    Z,
    X,
    weight,
    gamma,
    global_constraint,
    sakoe_chiba_radius,
    itakura_max_slope
):
    """Compute objective value and grad at Z for a given timeseries."""
    mask = _njit_compute_mask(Z.shape[0], X.shape[0], global_constraint, sakoe_chiba_radius, itakura_max_slope)
    D = cdist(Z, X) ** 2
    D[~mask] = numpy.inf
    return _njit_softdtw_func_from_distance_matrix(Z, X, D, weight, gamma, mask)


@njit(nogil=True)
def _njit_softdtw_func_from_distance_matrix(Z, X, D, weight, gamma, mask):
    """Compute objective value and grad at Z for a given distance matrix."""
    R = _njit_accumulated_matrix_from_distance_matrix(D, mask, gamma=gamma)
    E = _njit_soft_dtw_grad(D, R, gamma)
    return weight * R[-1, -1], weight * _njit_jacobian_product_sq_euc(Z, X, E)


@njit(nogil=True, parallel=True, fastmath=True)
def _njit_jacobian_product_sq_euc(X, Y, E):
    """Compute the square Euclidean product between the Jacobian
    (a linear map from m x d to m x n) and a matrix E.
    """

    m = X.shape[0]
    n = Y.shape[0]
    d = X.shape[1]
    G = numpy.zeros((m, d))

    for i in prange(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])

    return G


def softdtw_barycenter(
    X,
    gamma=1.0,
    weights=None,
    tol=1e-3,
    max_iter=50,
    init=None,
    n_jobs=None,
    **metric_params
):
    """Compute barycenter (time series averaging) under the soft-DTW
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
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.
    metric_params: dict or None (default: None)
        Soft-DTW constraint parameters to be used.
        See :ref:`tslearn.metrics.soft_dtw <fun-tslearn.metrics.soft_dtw>` for
        a list of accepted parameters
        If None, no constraint is used for soft-DTW computations.

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
    backend = instantiate_backend(X)
    X_ = to_time_series_dataset(X, be=backend)

    weights = _set_weights(weights, X_.shape[0])
    if init is None:
        if _check_equal_size(X_):
            barycenter = numpy.average(X_, axis=0, weights=weights)
        else:
            resampled_X = TimeSeriesResampler().fit_transform(X_)
            barycenter = numpy.average(resampled_X, axis=0, weights=weights)
    else:
        barycenter = init

    metric_params['global_constraint'] = GLOBAL_CONSTRAINT_CODE[metric_params.get('global_constraint')]

    if max_iter > 0:
        X_ = [_to_time_series(d, True, backend) for d in X_]

        def f(Z):
            return _acc_softdtw_func(Z, X_, weights, gamma, n_jobs=n_jobs, **metric_params)

        # The function works with vectors so we need to vectorize barycenter.
        res = minimize(f, barycenter.ravel(), method="L-BFGS-B", tol=tol, jac=True,
                       options=dict(maxiter=max_iter))
        return res.x.reshape(barycenter.shape)
    else:
        return barycenter
