STUFF_cysax = "cysax"

import numpy
from sklearn.linear_model import LinearRegression

cimport numpy
cimport cython

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

DTYPE = numpy.float
DTYPE_INT = numpy.int
ctypedef numpy.float_t DTYPE_t
ctypedef numpy.int_t DTYPE_INT_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def inv_transform_paa(numpy.ndarray[DTYPE_t, ndim=3] dataset_paa, int original_size):
    cdef int n_ts = dataset_paa.shape[0]
    cdef int sz = dataset_paa.shape[1]
    cdef int d = dataset_paa.shape[2]
    cdef int i = 0
    cdef int t = 0
    cdef int di = 0
    cdef int t0 = 0
    cdef int seg_sz = original_size // sz
    cdef numpy.ndarray[DTYPE_t, ndim=3] dataset_out = numpy.zeros((n_ts, original_size, d))

    for i in range(n_ts):
        for t in range(sz):
            t0 = t * seg_sz
            for di in range(d):
                dataset_out[i, t0:t0+seg_sz, di] = dataset_paa[i, t, di]
    return dataset_out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cydist_sax(numpy.ndarray[DTYPE_INT_t, ndim=2] sax1, numpy.ndarray[DTYPE_INT_t, ndim=2] sax2,
               numpy.ndarray[DTYPE_t, ndim=1] breakpoints, int original_size):
    assert sax1.shape[0] == sax2.shape[0] and sax1.shape[1] == sax2.shape[1]
    cdef DTYPE_t s = 0.
    cdef int sz = sax1.shape[0]
    cdef int d = sax1.shape[1]
    cdef int t = 0
    cdef int di = 0
    for t in range(sz):
        for di in range(d):
            if numpy.abs(sax1[t, di] - sax2[t, di]) > 1:
                max_symbol = max(sax1[t, di], sax2[t, di])
                min_symbol = min(sax1[t, di], sax2[t, di])
                s += (breakpoints[max_symbol - 1] - breakpoints[min_symbol]) ** 2
    return numpy.sqrt(s * float(original_size) / sz)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def inv_transform_sax(numpy.ndarray[DTYPE_INT_t, ndim=3] dataset_sax,
                      numpy.ndarray[DTYPE_t, ndim=1] breakpoints_middle_, int original_size):
    cdef int n_ts = dataset_sax.shape[0]
    cdef int sz = dataset_sax.shape[1]
    cdef int d = dataset_sax.shape[2]
    cdef int i = 0
    cdef int t = 0
    cdef int di = 0
    cdef int t0 = 0
    cdef int seg_sz = original_size // sz
    cdef numpy.ndarray[DTYPE_t, ndim=3] dataset_out = numpy.zeros((n_ts, original_size, d))

    for i in range(n_ts):
        for t in range(sz):
            t0 = t * seg_sz
            for di in range(d):
                dataset_out[i, t0:t0+seg_sz, di] = breakpoints_middle_[dataset_sax[i, t, di]]
    return dataset_out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cyslopes(numpy.ndarray[DTYPE_t, ndim=3] dataset, int t0):
    cdef int i = 0
    cdef int d = 0
    cdef int sz = dataset.shape[1]
    cdef numpy.ndarray[DTYPE_t, ndim=2] dataset_out = numpy.empty((dataset.shape[0], dataset.shape[2]))
    cdef numpy.ndarray[DTYPE_INT_t, ndim=2] vec_t = numpy.arange(t0, t0 + sz).reshape((-1, 1))

    for i in range(dataset.shape[0]):
        for d in range(dataset.shape[2]):
            dataset_out[i, d] = LinearRegression().fit(vec_t, dataset[i, :, d].reshape((-1, 1))).coef_[0]
    return dataset_out


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cydist_1d_sax(numpy.ndarray[DTYPE_INT_t, ndim=2] sax1, numpy.ndarray[DTYPE_INT_t, ndim=2] sax2,
                  numpy.ndarray[DTYPE_t, ndim=1] breakpoints_avg_middle_,
                  numpy.ndarray[DTYPE_t, ndim=1] breakpoints_slope_middle_, int original_size):
    assert sax1.shape[0] == sax2.shape[0] and sax1.shape[1] == sax2.shape[1]
    cdef DTYPE_t s = 0.
    cdef int sz = sax1.shape[0]
    cdef int d = sax1.shape[1] // 2
    cdef int t = 0
    cdef int di = 0
    cdef int t0 = 0
    cdef int seg_sz = original_size // sz
    cdef DTYPE_t t_middle = 0.
    cdef DTYPE_t slope1 = 0.
    cdef DTYPE_t slope2 = 0.
    cdef DTYPE_t avg1 = 0.
    cdef DTYPE_t avg2 = 0.

    for t in range(sz):
        t0 = t * seg_sz
        t_middle = float(t0) + .5 * seg_sz
        for di in range(d):
            avg1 = breakpoints_avg_middle_[sax1[t, di]]
            avg2 = breakpoints_avg_middle_[sax2[t, di]]
            slope1 = breakpoints_slope_middle_[sax1[t, di + d]]
            slope2 = breakpoints_slope_middle_[sax2[t, di + d]]
            for tt in range(t0, seg_sz * (t + 1)):
                s += (avg1 + slope1 * (tt - t_middle) - (avg2 + slope2 * (tt - t_middle))) ** 2
    return numpy.sqrt(s)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def inv_transform_1d_sax(numpy.ndarray[DTYPE_INT_t, ndim=3] dataset_sax,
                         numpy.ndarray[DTYPE_t, ndim=1] breakpoints_avg_middle_,
                         numpy.ndarray[DTYPE_t, ndim=1] breakpoints_slope_middle_, int original_size):
    cdef int n_ts = dataset_sax.shape[0]
    cdef int sz = dataset_sax.shape[1]
    cdef int d = dataset_sax.shape[2] // 2
    cdef int i = 0
    cdef int t = 0
    cdef int di = 0
    cdef int t0 = 0
    cdef int seg_sz = original_size // sz
    cdef DTYPE_t t_middle = 0.
    cdef DTYPE_t slope = 0.
    cdef DTYPE_t avg = 0.
    cdef numpy.ndarray[DTYPE_t, ndim=3] dataset_out = numpy.empty((n_ts, original_size, d))

    for i in range(n_ts):
        for t in range(sz):
            t0 = t * seg_sz
            t_middle = float(t0) + .5 * seg_sz
            for di in range(d):
                avg = breakpoints_avg_middle_[dataset_sax[i, t, di]]
                slope = breakpoints_slope_middle_[dataset_sax[i, t, di + d]]
                for tt in range(t0, seg_sz * (t + 1)):
                    dataset_out[i, tt, di] = avg + slope * (tt - t_middle)
    return dataset_out
