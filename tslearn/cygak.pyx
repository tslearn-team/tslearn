STUFF_cygak = "cygak"

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
def gak(numpy.ndarray[DTYPE_t, ndim=2] s1, numpy.ndarray[DTYPE_t, ndim=2] s2, DTYPE_t sigma):
    """k = gak(s1, s2, sigma)
    Compute Global Alignment Kernel between (possibly multidimensional) time series and return it.
    Time series must be 2d numpy arrays of shape (size, dim). It is not required that both time series share the same
    length, but they must be the same dimension. dtype of the arrays must be numpy.float."""
    assert s1.dtype == DTYPE and s2.dtype == DTYPE
    cdef int l1 = s1.shape[0]
    cdef int l2 = s2.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] gram = - cdist(s1, s2, "sqeuclidean").astype(DTYPE) / (2 * sigma ** 2)
    cdef numpy.ndarray[DTYPE_t, ndim=2] cum_sum = numpy.zeros((l1 + 1, l2 + 1), dtype=DTYPE)

    cum_sum[0, 0] = 1.

    gram -= numpy.log(2 - numpy.exp(gram))
    gram = numpy.exp(gram)

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = (cum_sum[i, j + 1] + cum_sum[i + 1, j] + cum_sum[i, j]) * gram[i, j]

    return cum_sum[l1, l2]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def normalized_gak(numpy.ndarray[DTYPE_t, ndim=2] s1, numpy.ndarray[DTYPE_t, ndim=2] s2, DTYPE_t sigma):
    """k = normalized_gak(s1, s2, sigma)
    Compute normalized Global Alignment Kernel between (possibly multidimensional) time series and return it.
    Time series must be 2d numpy arrays of shape (size, dim). It is not required that both time series share the same
    length, but they must be the same dimension. dtype of the arrays must be numpy.float."""
    cdef DTYPE_t kij = gak(s1, s2, sigma)
    cdef DTYPE_t kii = gak(s1, s1, sigma)
    cdef DTYPE_t kjj = gak(s2, s2, sigma)

    return kij / numpy.sqrt(kii * kjj)  # TODO: deal with divide by 0 issue

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cdist_gak(numpy.ndarray[DTYPE_t, ndim=3] dataset1, numpy.ndarray[DTYPE_t, ndim=3] dataset2, DTYPE_t sigma,
              bool self_similarity):
    assert dataset1.dtype == DTYPE and dataset2.dtype == DTYPE
    cdef int n1 = dataset1.shape[0]
    cdef int n2 = dataset2.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] cross_dist = numpy.empty((n1, n2), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=1] kiis = numpy.empty((n1, ), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=1] kjjs = numpy.empty((n2, ), dtype=DTYPE)

    for i in range(n1):
        kiis[i] = gak(dataset1[i], dataset1[i], sigma)
    for j in range(n2):
        kjjs[j] = gak(dataset2[j], dataset2[j], sigma)

    kiis = numpy.sqrt(kiis)
    kjjs = numpy.sqrt(kjjs)

    for i in range(n1):
        for j in range(n2):
            if self_similarity and j < i:
                cross_dist[i, j] = cross_dist[j, i]
            elif self_similarity and i == j:
                cross_dist[i, j] = 1.
            else:
                cross_dist[i, j] = gak(dataset1[i], dataset2[j], sigma) / (kiis[i] * kjjs[j])  # TODO: deal with divide by 0 issue

    return cross_dist