STUFF_cycc = "cycc"

import numpy

cimport numpy
cimport cython
from cpython cimport bool

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def normalized_cc(numpy.ndarray[DTYPE_t, ndim=2] s1, numpy.ndarray[DTYPE_t, ndim=2] s2,
                  float norm1=-1., float norm2=-1.):
    assert s1.dtype == DTYPE and s2.dtype == DTYPE
    assert s1.shape[1] == s2.shape[1]
    cdef DTYPE_t s = 0.
    cdef int sz = s1.shape[0]
    cdef int d = s1.shape[1]
    # Compute fft size based on tip from
    # https://stackoverflow.com/questions/14267555/how-can-i-find-the-smallest-power-of-2-greater-than-n-in-python
    cdef int fft_sz = 1 << (2 * sz - 1).bit_length()
    cdef float denom = 0.
    cdef numpy.ndarray[DTYPE_t, ndim=2] cc

    if norm1 < 0.:
        norm1 = numpy.linalg.norm(s1)
    if norm2 < 0.:
        norm2 = numpy.linalg.norm(s2)

    denom = norm1 * norm2
    if denom < 1e-9:  # To avoid NaNs
        denom = numpy.inf

    cc = numpy.real(numpy.fft.ifft(numpy.fft.fft(s1, fft_sz, axis=0) *
                                   numpy.conj(numpy.fft.fft(s2, fft_sz, axis=0)), axis=0))
    cc = numpy.vstack((cc[-(sz-1):], cc[:sz]))
    return numpy.real(cc).sum(axis=-1) / denom

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cdist_normalized_cc(numpy.ndarray[DTYPE_t, ndim=3] dataset1, numpy.ndarray[DTYPE_t, ndim=3] dataset2,
                        numpy.ndarray[DTYPE_t, ndim=1] norms1, numpy.ndarray[DTYPE_t, ndim=1] norms2,
                        bool self_similarity):
    assert dataset1.dtype == DTYPE and dataset2.dtype == DTYPE
    assert dataset1.shape[2] == dataset2.shape[2]
    cdef int i = 0
    cdef int j = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] dists = numpy.empty((dataset1.shape[0], dataset2.shape[0]))

    if (norms1 < 0.).any():
        norms1 = numpy.linalg.norm(dataset1, axis=(1, 2))
    if (norms2 < 0.).any():
        norms2 = numpy.linalg.norm(dataset2, axis=(1, 2))

    for i in range(dataset1.shape[0]):
        for j in range(dataset2.shape[0]):
            if self_similarity and j < i:
                dists[i, j] = dists[j, i]
            elif self_similarity and i == j:
                dists[i, j] = 0.
            else:
                dists[i, j] = normalized_cc(dataset1[i], dataset2[j], norm1=norms1[i], norm2=norms2[j]).max()
    return dists


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def y_shifted_sbd_vec(numpy.ndarray[DTYPE_t, ndim=2] ref_ts, numpy.ndarray[DTYPE_t, ndim=3] dataset, float norm_ref,
                      numpy.ndarray[DTYPE_t, ndim=1] norms_dataset):
    assert dataset.dtype == DTYPE and ref_ts.dtype == DTYPE
    assert dataset.shape[1] == ref_ts.shape[0] and dataset.shape[2] == ref_ts.shape[1]
    cdef int i = 0
    cdef int sz = dataset.shape[1]
    cdef numpy.ndarray[DTYPE_t, ndim=3] dataset_shifted = numpy.zeros((dataset.shape[0], dataset.shape[1],
                                                                       dataset.shape[2]))

    if norm_ref < 0:
        norm_ref = numpy.linalg.norm(ref_ts)
    if (norms_dataset < 0.).any():
        norms_dataset = numpy.linalg.norm(dataset, axis=(1, 2))

    for i in range(dataset.shape[0]):
        cc = normalized_cc(ref_ts, dataset[i], norm1=norm_ref, norm2=norms_dataset[i])
        idx = numpy.argmax(cc)
        shift = idx - sz
        if shift > 0:
            #print(dataset_shifted[i, shift:].shape[0])
            #print(dataset[i, :-shift, :].shape[0])
            dataset_shifted[i, shift:] = dataset[i, :-shift, :]
        elif shift < 0:
            #print("Negative shift")
            #print(dataset_shifted[i, :shift].shape[0])
            #print(dataset[i, -shift:, :].shape[0])
            dataset_shifted[i, :shift] = dataset[i, -shift:, :]
        else:
            dataset_shifted[i] = dataset[i]
    return dataset_shifted