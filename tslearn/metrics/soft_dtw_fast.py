# Author: Mathieu Blondel
# License: Simplified BSD

# encoding: utf-8

import numpy as np
from numba import jit, njit, prange, float64

DBL_MAX = np.finfo("double").max


"""njit --> Ok

tslearn/metrics/soft_dtw_fast.py:73:            R[i, j] = D[i - 1, j - 1] + _softmin3(
"""


# @njit(parallel=True)
@njit(float64(float64, float64, float64, float64))
def _softmin3(a, b, c, gamma):
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


"""njit --> Ok

tslearn/metrics/softdtw_variants.py:653:        _soft_dtw(self.D, self.R_, gamma=self.gamma)
"""


# @njit(parallel=True)
@njit((float64[:, :], float64[:, :], float64))
def _soft_dtw(D, R, gamma):
    """Compute soft dynamic time warping.

    Parameters
    ----------
    D : array-like, shape=[m, n]
    R : array-like, shape=[m+2, n+2]
    gamma : float64
    """
    m = D.shape[0]
    n = D.shape[1]

    # Initialization.
    # R = np.zeros([m + 2, n + 2], dtype=np.float64)
    R[: m + 1, 0] = DBL_MAX
    R[0, : n + 1] = DBL_MAX
    R[0, 0] = 0

    # DP recursion.
    for i in prange(1, m + 1):
        for j in range(1, n + 1):
            # D is indexed starting from 0.
            R[i, j] = D[i - 1, j - 1] + _softmin3(
                R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma
            )


"""njit --> Ok

tslearn/metrics/softdtw_variants.py:682:        _soft_dtw_grad(D, self.R_, E, gamma=self.gamma)
"""


# @njit(parallel=True)
@njit((float64[:, :], float64[:, :], float64[:, :], float64))
def _soft_dtw_grad(D, R, E, gamma):
    """Compute gradient of soft-DTW w.r.t. D.

    Parameters
    ----------
    D : array-like, shape=[m, n]
    R : array-like, shape=[m+2, n+2]
    E : array-like, shape=[m+2, n+2]
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

    # DP recursion.
    for j in prange(n, 0, -1):  # ranges from n to 1
        for i in range(m, 0, -1):  # ranges from m to 1
            a = np.exp((R[i + 1, j] - R[i, j] - D[i, j - 1]) / gamma)
            b = np.exp((R[i, j + 1] - R[i, j] - D[i - 1, j]) / gamma)
            c = np.exp((R[i + 1, j + 1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c


"""njit --> Ok

tslearn/metrics/softdtw_variants.py:735:        _jacobian_product_sq_euc(self.X, self.Y, E.astype(numpy.float64), G)
"""


# @njit(parallel=True)
@njit((float64[:, :], float64[:, :], float64[:, :], float64[:, :]))
def _jacobian_product_sq_euc(X, Y, E, G):
    """Compute the square Euclidean product between the Jacobian
    (a linear map from m x d to m x n) and a matrix E.

    Parameters
    ----------
    X: array, shape = [m, d]
        First time series.
    Y: array, shape = [n, d]
        Second time series.
    E: array, shape = [m, n]
    G: array, shape = [m, d]
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
