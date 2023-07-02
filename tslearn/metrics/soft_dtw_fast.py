# Author: Mathieu Blondel
# License: Simplified BSD

# encoding: utf-8

import numpy as np
from numba import njit, prange

from tslearn.backend.backend import instantiate_backend

DBL_MAX = np.finfo("double").max


@njit(fastmath=True)
def _njit_softmin3(a, b, c, gamma):
    """Compute softmin of 3 input variables with parameter gamma.

    Parameters
    ----------
    a : float64
    b : float64
    c : float64
    gamma : float64

    Returns
    -------
    softmin_value : float64
    """
    a /= -gamma
    b /= -gamma
    c /= -gamma

    max_val = max(max(a, b), c)

    tmp = 0
    tmp += np.exp(a - max_val)
    tmp += np.exp(b - max_val)
    tmp += np.exp(c - max_val)
    softmin_value = -gamma * (np.log(tmp) + max_val)
    return softmin_value


def _softmin3(a, b, c, gamma, be=None):
    """Compute softmin of 3 input variables with parameter gamma.

    Parameters
    ----------
    a : float64
    b : float64
    c : float64
    gamma : float64
    be : Backend object or string or None
        Backend.

    Returns
    -------
    softmin_value : float64
    """
    be = instantiate_backend(be, a)

    a /= -gamma
    b /= -gamma
    c /= -gamma

    max_val = max(max(a, b), c)

    tmp = 0
    tmp += be.exp(a - max_val)
    tmp += be.exp(b - max_val)
    tmp += be.exp(c - max_val)
    softmin_value = -gamma * (be.log(tmp) + max_val)
    return softmin_value


@njit(fastmath=True)
def _njit_soft_dtw(D, R, gamma):
    """Compute soft dynamic time warping.

    Parameters
    ----------
    D : array-like, shape=[m, n], dtype=float64
    R : array-like, shape=[m+2, n+2], dtype=float64
    gamma : float64
    """
    m = D.shape[0]
    n = D.shape[1]

    # Initialization.
    R[: m + 1, 0] = DBL_MAX
    R[0, : n + 1] = DBL_MAX
    R[0, 0] = 0

    # DP recursion.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # D is indexed starting from 0.
            R[i, j] = D[i - 1, j - 1] + _njit_softmin3(
                R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma
            )


def _soft_dtw(D, R, gamma, be=None):
    """Compute soft dynamic time warping.

    Parameters
    ----------
    D : array-like, shape=[m, n], dtype=float64
    R : array-like, shape=[m+2, n+2], dtype=float64
    gamma : float64
    be : Backend object or string or None
        Backend.
    """
    be = instantiate_backend(be, D)

    m = D.shape[0]
    n = D.shape[1]

    # Initialization.
    R[: m + 1, 0] = be.dbl_max
    R[0, : n + 1] = be.dbl_max
    R[0, 0] = 0

    # DP recursion.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # D is indexed starting from 0.
            R[i, j] = D[i - 1, j - 1] + _softmin3(
                be.copy(R[i - 1, j]),
                be.copy(R[i - 1, j - 1]),
                be.copy(R[i, j - 1]),
                gamma,
                be=be,
            )


@njit(parallel=True)
def _njit_soft_dtw_batch(D, R, gamma):
    """Compute soft dynamic time warping.

    Parameters
    ----------
    D : array-like, shape=[b, m, n], dtype=float64
    R : array-like, shape=[b, m+2, n+2], dtype=float64
    gamma : float64
    """
    for i_sample in prange(D.shape[0]):
        _njit_soft_dtw(D[i_sample, :, :], R[i_sample, :, :], gamma)


def _soft_dtw_batch(D, R, gamma, be=None):
    """Compute soft dynamic time warping.

    Parameters
    ----------
    D : array-like, shape=[b, m, n], dtype=float64
    R : array-like, shape=[b, m+2, n+2], dtype=float64
    gamma : float64
    be : Backend object or string or None
        Backend.
    """
    be = instantiate_backend(be, D)
    for i_sample in range(D.shape[0]):
        _soft_dtw(D[i_sample, :, :], R[i_sample, :, :], gamma, be=be)


@njit(fastmath=True)
def _njit_soft_dtw_grad(D, R, E, gamma):
    """Compute gradient of soft-DTW w.r.t. D.

    Parameters
    ----------
    D : array-like, shape=[m, n], dtype=float64
    R : array-like, shape=[m+2, n+2], dtype=float64
    E : array-like, shape=[m+2, n+2], dtype=float64
    gamma : float64
    """
    m = D.shape[0] - 1
    n = D.shape[1] - 1

    # Initialization.
    D[:m, n] = 0
    D[m, :n] = 0
    R[1 : m + 1, n + 1] = -DBL_MAX
    R[m + 1, 1 : n + 1] = -DBL_MAX

    E[m + 1, n + 1] = 1
    R[m + 1, n + 1] = R[m, n]
    D[m, n] = 0

    for j in range(n, 0, -1):  # ranges from n to 1
        for i in range(m, 0, -1):  # ranges from m to 1
            a = np.exp((R[i + 1, j] - R[i, j] - D[i, j - 1]) / gamma)
            b = np.exp((R[i, j + 1] - R[i, j] - D[i - 1, j]) / gamma)
            c = np.exp((R[i + 1, j + 1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c


def _soft_dtw_grad(D, R, E, gamma, be=None):
    """Compute gradient of soft-DTW w.r.t. D.

    Parameters
    ----------
    D : array-like, shape=[m, n], dtype=float64
    R : array-like, shape=[m+2, n+2], dtype=float64
    E : array-like, shape=[m+2, n+2], dtype=float64
    gamma : float64
    be : Backend object or string or None
        Backend.
    """
    be = instantiate_backend(be, D)

    m = D.shape[0] - 1
    n = D.shape[1] - 1

    # Initialization.
    D[:m, n] = 0
    D[m, :n] = 0
    R[1 : m + 1, n + 1] = -be.dbl_max
    R[m + 1, 1 : n + 1] = -be.dbl_max

    E[m + 1, n + 1] = 1
    R[m + 1, n + 1] = R[m, n]
    D[m, n] = 0

    for j in range(n, 0, -1):  # ranges from n to 1
        for i in range(m, 0, -1):  # ranges from m to 1
            a = be.exp((R[i + 1, j] - R[i, j] - D[i, j - 1]) / gamma)
            b = be.exp((R[i, j + 1] - R[i, j] - D[i - 1, j]) / gamma)
            c = be.exp((R[i + 1, j + 1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c


@njit(parallel=True)
def _njit_soft_dtw_grad_batch(D, R, E, gamma):
    """Compute gradient of soft-DTW w.r.t. D.

    Parameters
    ----------
    D : array-like, shape=[b, m, n], dtype=float64
    R : array-like, shape=[b, m+2, n+2], dtype=float64
    E : array-like, shape=[b, m+2, n+2], dtype=float64
    gamma : float64
    """
    for i_sample in prange(D.shape[0]):
        _njit_soft_dtw_grad(D[i_sample, :, :], R[i_sample, :, :], E[i_sample, :, :], gamma)


def _soft_dtw_grad_batch(D, R, E, gamma, be=None):
    """Compute gradient of soft-DTW w.r.t. D.

    Parameters
    ----------
    D : array-like, shape=[b, m, n], dtype=float64
    R : array-like, shape=[b, m+2, n+2], dtype=float64
    E : array-like, shape=[b, m+2, n+2], dtype=float64
    gamma : float64
    be : Backend object or string or None
        Backend.
    """
    be = instantiate_backend(be, D)
    for i_sample in prange(D.shape[0]):
        _soft_dtw_grad(D[i_sample, :, :], R[i_sample, :, :], E[i_sample, :, :], gamma, be=be)


@njit(parallel=True, fastmath=True)
def _njit_jacobian_product_sq_euc(X, Y, E, G):
    """Compute the square Euclidean product between the Jacobian
    (a linear map from m x d to m x n) and a matrix E.

    Parameters
    ----------
    X: array, shape = [m, d], dtype=float64
        First time series.
    Y: array, shape = [n, d], dtype=float64
        Second time series.
    E: array, shape = [m, n], dtype=float64
    G: array, shape = [m, d], dtype=float64
        Product with Jacobian
        ([m x d, m x n] * [m x n] = [m x d]).
    """
    m = X.shape[0]
    n = Y.shape[0]
    d = X.shape[1]

    for i in prange(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])


def _jacobian_product_sq_euc(X, Y, E, G):
    """Compute the square Euclidean product between the Jacobian
    (a linear map from m x d to m x n) and a matrix E.

    Parameters
    ----------
    X: array, shape = [m, d], dtype=float64
        First time series.
    Y: array, shape = [n, d], dtype=float64
        Second time series.
    E: array, shape = [m, n], dtype=float64
    G: array, shape = [m, d], dtype=float64
        Product with Jacobian
        ([m x d, m x n] * [m x n] = [m x d]).
    """
    m = X.shape[0]
    n = Y.shape[0]
    d = X.shape[1]

    for i in range(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])
