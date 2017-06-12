STUFF_cylrdtw = "cylrdtw"

import numpy
from scipy.spatial.distance import cdist

cimport numpy
cimport cython
from cpython cimport bool

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_probas_formula(DTYPE_t cost_up, DTYPE_t cost_right, DTYPE_t cost_diagonal, DTYPE_t gamma):
    cdef DTYPE_t min_val
    cdef DTYPE_t p_up = 0.
    cdef DTYPE_t p_right = 0.
    cdef DTYPE_t p_diag = 0.
    cdef DTYPE_t s
    if gamma < 1e-12:
        min_val = min(cost_up, cost_right, cost_diagonal)
        if cost_up == min_val:
            p_up = 1.
        if cost_diagonal == min_val:
            p_diag = 1.
        if cost_right == min_val:
            p_right = 1.
        s = p_up + p_right + p_diag
        return p_up / s, p_right / s, p_diag / s

    p_up = 1. / 3. * (1. + (cost_right + cost_diagonal - 2. * cost_up) / (2. * gamma))
    p_right = 1. / 3. * (1. + (cost_up + cost_diagonal - 2. * cost_right) / (2. * gamma))
    if p_up >= 0. and p_right >= 0. and p_up + p_right <= 1.:
        p_diag = 1. - p_up - p_right
    elif p_up < 0.:
        p_right = .5 * (1. + (cost_diagonal - cost_right) / (2. * gamma))
        if 0. <= p_right <= 1.:
            p_up = 0.
            p_diag = 1. - p_right
        elif p_right < 0.:
            p_up = 0.
            p_right = 0.
            p_diag = 1.
        else:
            p_up, p_right, p_diag = 0., 1., 0.
    elif p_right < 0.:
        p_up = .5 * (1. + (cost_diagonal - cost_up) / (2 * gamma))
        if 0. <= p_up <= 1.:
            p_right, p_diag = 0., 1. - p_up
        elif p_up < 0.:
            p_up, p_right, p_diag = 0., 0., 1.
        else:
            p_up, p_right, p_diag = 1., 0., 0.
    else:
        p_up = .5 * (1. + (cost_right - cost_up) / (2 * gamma))
        if 0. <= p_up <= 1.:
            p_right, p_diag = 1. - p_up, 0.
        elif p_up < 0.:
            p_up, p_right, p_diag = 0., 1., 0.
        else:
            p_up, p_right, p_diag = 1., 0., 0.
    return p_up, p_right, p_diag


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def lr_dtw(numpy.ndarray[DTYPE_t, ndim=2] s1, numpy.ndarray[DTYPE_t, ndim=2] s2, DTYPE_t gamma):
    cdef numpy.ndarray[DTYPE_t, ndim=2] mat_dist = cdist(s1, s2, "sqeuclidean").astype(DTYPE)
    cdef int n = mat_dist.shape[0]
    cdef int m = mat_dist.shape[1]
    cdef numpy.ndarray[DTYPE_t, ndim=2] mat_cost = numpy.zeros((n, m), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=3] probas = numpy.zeros((n, m, 3), dtype=DTYPE)
    cdef int i
    cdef int j
    cdef int UP = 0
    cdef int RIGHT = 1
    cdef int DIAGONAL = 2
    cdef DTYPE_t p_up
    cdef DTYPE_t p_right
    cdef DTYPE_t p_diag

    probas[1:, 0, UP] = 1.
    probas[0, 1:, RIGHT] = 1.
    for i in range(1, n):
        mat_cost[i, 0] = mat_cost[i - 1, 0] + mat_dist[i, 0]
    for j in range(1, m):
        mat_cost[0, j] = mat_cost[0, j - 1] + mat_dist[0, j]
    for i in range(1, n):
        for j in range(1, m):
            p_up, p_right, p_diag = get_probas_formula(mat_cost[i - 1, j], mat_cost[i, j - 1], mat_cost[i - 1, j - 1],
                                                       gamma)
            probas[i, j, UP] = p_up
            probas[i, j, RIGHT] = p_right
            probas[i, j, DIAGONAL] = p_diag
            mat_cost[i, j] = (probas[i, j, UP] * mat_cost[i - 1, j]
                              + probas[i, j, RIGHT] * mat_cost[i, j - 1]
                              + probas[i, j, DIAGONAL] * mat_cost[i - 1, j - 1]
                              + mat_dist[i, j])
    return numpy.sqrt(mat_cost[n - 1, m - 1]), probas


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def lr_dtw_backtrace(numpy.ndarray[DTYPE_t, ndim=3] probas):
    cdef int n = probas.shape[0]
    cdef int m = probas.shape[1]
    cdef i
    cdef j
    cdef numpy.ndarray[DTYPE_t, ndim=2] mat_probas = numpy.zeros((n, m), dtype=DTYPE)
    mat_probas[n - 1, m - 1] = 1.
    cdef int UP = 0
    cdef int RIGHT = 1
    cdef int DIAGONAL = 2

    for i in range(n - 2, -1, -1):
        mat_probas[i, m - 1] = mat_probas[i + 1, m - 1] * probas[i + 1, m - 1, UP]
    for j in range(m - 2, -1, -1):
        mat_probas[n - 1, j] = mat_probas[n - 1, j + 1] * probas[n - 1, j + 1, RIGHT]
    for i in range(n - 2, -1, -1):
        for j in range(m - 2, -1, -1):
            mat_probas[i, j] = mat_probas[i + 1, j] * probas[i + 1, j, UP] + \
                               mat_probas[i, j + 1] * probas[i, j + 1, RIGHT] + \
                               mat_probas[i + 1, j + 1] * probas[i + 1, j + 1, DIAGONAL]
    return mat_probas

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cdist_lr_dtw(numpy.ndarray[DTYPE_t, ndim=3] dataset1, numpy.ndarray[DTYPE_t, ndim=3] dataset2, DTYPE_t gamma,
                bool self_similarity):
    assert dataset1.dtype == DTYPE and dataset2.dtype == DTYPE
    cdef int n1 = dataset1.shape[0]
    cdef int n2 = dataset2.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] cross_dist = numpy.empty((n1, n2), dtype=DTYPE)

    for i in range(n1):
        for j in range(n2):
            if self_similarity and j < i:
                cross_dist[i, j] = cross_dist[j, i]
            elif self_similarity and i == j:
                cross_dist[i, j] = 0.
            else:
                cross_dist[i, j] = lr_dtw(dataset1[i], dataset2[j], gamma)[0]

    return cross_dist