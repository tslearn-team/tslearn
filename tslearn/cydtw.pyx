STUFF_cydtw = "cydtw"

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
def dtw_path(numpy.ndarray[DTYPE_t, ndim=2] s1, numpy.ndarray[DTYPE_t, ndim=2] s2):
    """path, sim = dtw_path(s1, s2)
    Compute DTW similarity measure between (possibly multidimensional) time series and return both the path and the similarity.
    Time series must be 2d numpy arrays of shape (size, dim). It is not required that both time series share the same
    length, but they must be the same dimension. dtype of the arrays must be numpy.float."""
    assert s1.dtype == DTYPE and s2.dtype == DTYPE
    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
    cdef int l1 = s1.shape[0]
    cdef int l2 = s2.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef int argmin_pred = -1
    cdef numpy.ndarray[DTYPE_t, ndim=2] cross_dist = cdist(s1, s2, "sqeuclidean").astype(DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] cum_sum = numpy.zeros((l1 + 1, l2 + 1), dtype=DTYPE)
    cum_sum[1:, 0] = numpy.inf
    cum_sum[0, 1:] = numpy.inf
    cdef numpy.ndarray[numpy.int_t, ndim=3] predecessors = numpy.zeros((l1, l2, 2), dtype=numpy.int) - 1
    cdef numpy.ndarray[DTYPE_t, ndim=1] candidates = numpy.zeros((3, ), dtype=DTYPE)
    cdef list best_path

    for i in range(l1):
        for j in range(l2):
            candidates[0] = cum_sum[i, j + 1]
            candidates[1] = cum_sum[i + 1, j]
            candidates[2] = cum_sum[i, j]
            if candidates[0] <= candidates[1] and candidates[0] <= candidates[2]:
                argmin_pred = 0
            elif candidates[1] <= candidates[2]:
                argmin_pred = 1
            else:
                argmin_pred = 2
            cum_sum[i + 1, j + 1] = candidates[argmin_pred] + cross_dist[i, j]
            if i + j > 0:
                if argmin_pred == 0:
                    predecessors[i, j, 0] = i - 1
                    predecessors[i, j, 1] = j
                elif argmin_pred == 1:
                    predecessors[i, j, 0] = i
                    predecessors[i, j, 1] = j - 1
                else:
                    predecessors[i, j, 0] = i - 1
                    predecessors[i, j, 1] = j - 1

    i = l1 - 1
    j = l2 - 1
    best_path = [(i, j)]
    while predecessors[i, j, 0] >= 0 and predecessors[i, j, 1] >= 0:
        i, j = predecessors[i, j, 0], predecessors[i, j, 1]
        best_path.insert(0, (i, j))

    return best_path, numpy.sqrt(cum_sum[l1, l2])

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def dtw(numpy.ndarray[DTYPE_t, ndim=2] s1, numpy.ndarray[DTYPE_t, ndim=2] s2):
    """sim = dtw(s1, s2)
    Compute DTW similarity measure between (possibly multidimensional) time series and return it.
    Time series must be 2d numpy arrays of shape (size, dim). It is not required that both time series share the same
    length, but they must be the same dimension. dtype of the arrays must be numpy.float."""
    assert s1.dtype == DTYPE and s2.dtype == DTYPE
    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
    cdef int l1 = s1.shape[0]
    cdef int l2 = s2.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] cross_dist = cdist(s1, s2, "sqeuclidean").astype(DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] cum_sum = numpy.zeros((l1 + 1, l2 + 1), dtype=DTYPE)
    cum_sum[1:, 0] = numpy.inf
    cum_sum[0, 1:] = numpy.inf

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = min(cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]) + cross_dist[i, j]

    return numpy.sqrt(cum_sum[l1, l2])

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cdist_dtw(numpy.ndarray[DTYPE_t, ndim=3] dataset1, numpy.ndarray[DTYPE_t, ndim=3] dataset2, bool self_similarity):
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
                cross_dist[i, j] = dtw(dataset1[i], dataset2[j])

    return cross_dist

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def lb_enveloppe(numpy.ndarray[DTYPE_t, ndim=2] time_series, int radius):
    assert time_series.dtype == DTYPE and time_series.shape[1] == 1
    cdef int sz = time_series.shape[0]
    cdef int i = 0
    cdef int min_idx = 0
    cdef int max_idx = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] enveloppe_up = numpy.empty((sz, 1), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] enveloppe_down = numpy.empty((sz, 1), dtype=DTYPE)

    for i in range(sz):
        min_idx = i - radius
        max_idx = i + radius
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        enveloppe_down[i, 0] = time_series[min_idx:max_idx, 0].min()
        enveloppe_up[i, 0] = time_series[min_idx:max_idx, 0].max()

    return enveloppe_down, enveloppe_up
