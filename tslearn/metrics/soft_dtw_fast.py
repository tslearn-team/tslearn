# Author: Mathieu Blondel
# License: Simplified BSD

# encoding: utf-8

from numba import njit, prange

from tslearn.backend.backend import Backend


def _softmin3(a, b, c, gamma, be=None):
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
    if be is None:
        be = Backend(a)
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


njit_softmin3 = njit(fastmath=True)(_softmin3)


def _soft_dtw(D, R, gamma, be=None):
    """Compute soft dynamic time warping.

    Parameters
    ----------
    D : array-like, shape=[m, n], dtype=float64
    R : array-like, shape=[m+2, n+2], dtype=float64
    gamma : float64
    """
    if be is None:
        be = Backend(D)

    if be.is_numpy:
        softmin3 = njit_softmin3
    else:
        softmin3 = _softmin3

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
            R[i, j] = D[i - 1, j - 1] + softmin3(
                R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma, be=be
            )


njit_soft_dtw = njit(parallel=True, fastmath=True)(_soft_dtw)


def _soft_dtw_grad(D, R, E, gamma, be=None):
    """Compute gradient of soft-DTW w.r.t. D.

    Parameters
    ----------
    D : array-like, shape=[m, n], dtype=float64
    R : array-like, shape=[m+2, n+2], dtype=float64
    E : array-like, shape=[m+2, n+2], dtype=float64
    gamma : float64
    """
    if be is None:
        be = Backend(D)

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


jit_soft_dtw_grad = njit(parallel=True, fastmath=True)(_soft_dtw_grad)


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

    for i in prange(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i, j] * 2 * (X[i, k] - Y[j, k])


njit_jacobian_product_sq_euc = njit(parallel=True, fastmath=True)(
    _jacobian_product_sq_euc
)


import numpy as np
from numba import njit
from tslearn.backend import Backend

x = np.array([1, 2, 3])
y = np.array([1, 2.2, -3])


@njit()
def addition(x, y):
    return np.add(x, y)


print(addition(x, y))


@njit()
def main_addition(backend, x, y):
    # if backend == np:
    #     print('backend == np')
    return np.add(x, y)
    return addition(x, y)


backend = Backend("numpy")

print(main_addition(backend, x, y))
print(main_addition(x, y))


# class TestNjit():
#     def __init__(self):
#         self.add = addition
#
#     # @staticmethod
#     # @njit()
#     # def add(x, y):
#     #     return np.add(x, y)
#
#
# backend = Backend("numpy")
# print(backend)
#
# @njit()
# def main_function(x, y, be=None):
#     print('be in main')
#     print(be)
#     print(be is None)
#     if be is None:
#         print('be is None')
#         be = Backend(x)
#     # return addition(x, y)
#     # test_njit = TestNjit()
#     # return test_njit.add(x, y)
#     return be.exp(x + y)
#
#
# a = TestNjit().add(x, y)
# print(a)
#
# b = main_function(x, y, backend)
# print(b)
