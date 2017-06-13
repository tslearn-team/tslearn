STUFF_cysax = "cysax"

import numpy
from scipy.spatial.distance import cdist

cimport numpy
cimport cython
from cpython cimport bool

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

DTYPE = numpy.float
DTYPE_INT = numpy.int
ctypedef numpy.float_t DTYPE_t
ctypedef numpy.int_t DTYPE_INT_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def distance_sax(numpy.ndarray[DTYPE_INT_t, ndim=2] s1, numpy.ndarray[DTYPE_INT_t, ndim=2] s2,
                 numpy.ndarray[DTYPE_t, ndim=1] breakpoints, int original_size):
    assert s1.dtype == DTYPE_INT and s2.dtype == DTYPE_INT
    assert s1.shape[0] == s2.shape[0] and s1.shape[1] == s2.shape[1]
    cdef DTYPE_t s = 0.
    cdef int sz = s1.shape[0]
    cdef int d = s1.shape[1]
    cdef int t = 0
    cdef int di = 0
    for t in range(sz):
        for di in range(d):
            if numpy.abs(s1[t, di] - s2[t, di]) > 1:
                max_symbol = max(s1[t, di], s2[t, di])
                min_symbol = min(s1[t, di], s2[t, di])
                s += (breakpoints[max_symbol - 1] - breakpoints[min_symbol]) ** 2
    return numpy.sqrt(s * float(original_size) / sz)
