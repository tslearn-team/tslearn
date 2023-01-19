# Author: Mathieu Blondel
# License: Simplified BSD

# encoding: utf-8

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
    elif isinstance(be, str):
        be = Backend(be)
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
    elif isinstance(be, str):
        be = Backend(be)

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
                R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma, be=be
            )


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
    elif isinstance(be, str):
        be = Backend(be)

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