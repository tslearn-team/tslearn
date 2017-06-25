# Author: Mathieu Blondel
# License: Simplified BSD

# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


from libc.float cimport DBL_MAX
from libc.math cimport exp, log
from libc.string cimport memset


cdef inline double _softmin3(DTYPE_t a,
                             DTYPE_t b,
                             DTYPE_t c,
                             DTYPE_t gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma

    cdef DTYPE_t max_val = max(max(a, b), c)

    cdef DTYPE_t tmp = 0
    tmp += exp(a - max_val)
    tmp += exp(b - max_val)
    tmp += exp(c - max_val)

    return -gamma * (log(tmp) + max_val)


def _soft_dtw(np.ndarray[DTYPE_t, ndim=2] D,
              np.ndarray[DTYPE_t, ndim=2] R,
              DTYPE_t gamma):

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    cdef int i, j

    # Initialization.
    memset(<void*>R.data, 0, (m+1) * (n+1) * sizeof(DTYPE_t))

    for i in range(m + 1):
        R[i, 0] = DBL_MAX

    for j in range(n + 1):
        R[0, j] = DBL_MAX

    R[0, 0] = 0

    # DP recursion.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # D is indexed starting from 0.
            R[i, j] = D[i-1, j-1] + _softmin3(R[i-1, j],
                                              R[i-1, j-1],
                                              R[i, j-1],
                                              gamma)


def _soft_dtw_grad(np.ndarray[DTYPE_t, ndim=2] D,
                   np.ndarray[DTYPE_t, ndim=2] R,
                   np.ndarray[DTYPE_t, ndim=2] E,
                   DTYPE_t gamma):

    # We added an extra row and an extra column on the Python side.
    cdef int m = D.shape[0] - 1
    cdef int n = D.shape[1] - 1

    cdef int i, j
    cdef DTYPE_t a, b, c

    # Initialization.
    memset(<void*>E.data, 0, (m+2) * (n+2) * sizeof(DTYPE_t))

    for i in range(1, m+1):
        # For D, indices start from 0 throughout.
        D[i-1, n] = 0
        R[i, n+1] = -DBL_MAX

    for j in range(1, n+1):
        D[m, j-1] = 0
        R[m+1, j] = -DBL_MAX

    E[m+1, n+1] = 1
    R[m+1, n+1] = R[m, n]
    D[m, n] = 0

    # DP recursion.
    for j in reversed(range(1, n+1)):  # ranges from n to 1
        for i in reversed(range(1, m+1)):  # ranges from m to 1
            a = exp((R[i+1, j] - R[i, j] - D[i, j-1]) / gamma)
            b = exp((R[i, j+1] - R[i, j] - D[i-1, j]) / gamma)
            c = exp((R[i+1, j+1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i+1, j] * a + E[i, j+1] * b + E[i+1,j+1] * c


def _jacobian_product_sq_euc(np.ndarray[DTYPE_t, ndim=2] X,
                             np.ndarray[DTYPE_t, ndim=2] Y,
                             np.ndarray[DTYPE_t, ndim=2] E,
                             np.ndarray[DTYPE_t, ndim=2] G):
    cdef int m = X.shape[0]
    cdef int n = Y.shape[0]
    cdef int d = X.shape[1]

    for i in range(m):
        for j in range(n):
            for k in range(d):
                G[i, k] += E[i,j] * 2 * (X[i, k] - Y[j, k])
