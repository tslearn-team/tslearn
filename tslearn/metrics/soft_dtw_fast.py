# Author: Mathieu Blondel
# License: Simplified BSD

# encoding: utf-8

import numpy as np
from numba import njit, prange

DTYPE = np.float64
DBL_MAX = np.finfo("double").max


@njit(parallel=True)
def _softmin3(a, b, c, gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma

    max_val = max(max(a, b), c)

    tmp = 0
    tmp += np.exp(a - max_val)
    tmp += np.exp(b - max_val)
    tmp += np.exp(c - max_val)

    return -gamma * (np.log(tmp) + max_val)


@njit(parallel=True)
def _soft_dtw(D, R, gamma):

    m = D.shape[0]
    n = D.shape[1]

    # Initialization.
    for i in prange(m + 1):
        R[i, 0] = DBL_MAX

    for j in prange(n + 1):
        R[0, j] = DBL_MAX

    R[0, 0] = 0

    # DP recursion.
    for i in prange(1, m + 1):
        for j in range(1, n + 1):
            # D is indexed starting from 0.
            R[i, j] = D[i - 1, j - 1] + _softmin3(
                R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma
            )


@njit(parallel=True)
def _soft_dtw_grad(D, R, E, gamma):

    # We added an extra row and an extra column on the Python side.
    m = D.shape[0] - 1
    n = D.shape[1] - 1

    # Initialization.
    for i in prange(1, m + 1):
        # For D, indices start from 0 throughout.
        D[i - 1, n] = 0
        R[i, n + 1] = -DBL_MAX

    for j in prange(1, n + 1):
        D[m, j - 1] = 0
        R[m + 1, j] = -DBL_MAX

    E[m + 1, n + 1] = 1
    R[m + 1, n + 1] = R[m, n]
    D[m, n] = 0

    # DP recursion.
    for k in prange(n):
        j = n - k  # ranges from n to 1
        for p in range(m):
            i = m - p  # ranges from m to 1
            a = np.exp((R[i + 1, j] - R[i, j] - D[i, j - 1]) / gamma)
            b = np.exp((R[i, j + 1] - R[i, j] - D[i - 1, j]) / gamma)
            c = np.exp((R[i + 1, j + 1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c


@njit(parallel=True)
def _jacobian_product_sq_euc(X, Y, E, G):
    m = X.shape[0]
    n = Y.shape[0]
    d = X.shape[1]

    for i in prange(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])
