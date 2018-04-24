STUFF_cydtw = "cydtw"

import numpy
from scipy.spatial.distance import cdist
# from tslearn.metrics import lb_keogh

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
def sakoe_chiba_mask(int sz1, int sz2, int radius):
    cdef int i = 0
    cdef int j = 0
    cdef DTYPE_t expected_j = 0.
    cdef DTYPE_t ratio = float(sz2 - 1) / (sz1 - 1)
    cdef numpy.ndarray[DTYPE_t, ndim=2] mask = numpy.zeros((sz1, sz2), dtype=DTYPE)

    for i in range(sz1):
        for j in range(sz2):
            expected_j = float(i) * ratio
            if abs(expected_j - j) > radius:
                mask[i, j] = numpy.inf
    return mask


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def itakura_mask(int sz1, int sz2):
    cdef int i = 0
    cdef int j = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] mask = numpy.zeros((sz1, sz2), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] mask_out = numpy.zeros((sz1, sz2), dtype=DTYPE)
    mask[:, 0:2] = numpy.inf
    mask[0:2, :] = numpy.inf
    mask[0, 0] = 0.
    mask[1, 1] = 0.
    mask[1, 2] = 0.
    mask[2, 1] = 0.

    for i in range(2, sz1):
        for j in range(2, sz2):
            if numpy.alltrue(~numpy.isfinite([mask[i-1, j-1], mask[i-2, j-1], mask[i-1, j-1]])):
                mask[i, j] = numpy.inf

    mask_out[~numpy.logical_and(numpy.isfinite(mask), numpy.isfinite(mask[::-1, ::-1]))] = numpy.inf
    return mask_out

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def ts_size(numpy.ndarray[DTYPE_t, ndim=2] ts):
    cdef int sz = ts.shape[0]
    while not numpy.any(numpy.isfinite(ts[sz - 1])):
        sz -= 1
    return sz


@cython.boundscheck(False)
@cython.wraparound(False)
def dtw_path(numpy.ndarray[DTYPE_t, ndim=2] s1, numpy.ndarray[DTYPE_t, ndim=2] s2, numpy.ndarray[DTYPE_t, ndim=2] mask):
    assert s1.dtype == DTYPE and s2.dtype == DTYPE

    cdef int l1 = ts_size(s1)
    cdef int l2 = ts_size(s2)
    assert l1 <= mask.shape[0] and l2 <= mask.shape[1]

    cdef int i = 0
    cdef int j = 0
    cdef int argmin_pred = -1
    cdef numpy.ndarray[DTYPE_t, ndim=2] cross_dist = cdist(s1[:l1], s2[:l2], "sqeuclidean").astype(DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] cum_sum = numpy.zeros((l1 + 1, l2 + 1), dtype=DTYPE)
    cum_sum[1:, 0] = numpy.inf
    cum_sum[0, 1:] = numpy.inf
    cdef numpy.ndarray[numpy.int_t, ndim=3] predecessors = numpy.zeros((l1, l2, 2), dtype=numpy.int) - 1
    cdef numpy.ndarray[DTYPE_t, ndim=1] candidates = numpy.zeros((3, ), dtype=DTYPE)
    cdef list best_path

    cross_dist[~numpy.isfinite(mask[:l1, :l2])] = numpy.inf

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = cross_dist[i, j]
            if numpy.isfinite(cum_sum[i + 1, j + 1]):
                candidates[0] = cum_sum[i, j + 1]
                candidates[1] = cum_sum[i + 1, j]
                candidates[2] = cum_sum[i, j]
                if candidates[0] <= candidates[1] and candidates[0] <= candidates[2]:
                    argmin_pred = 0
                elif candidates[1] <= candidates[2]:
                    argmin_pred = 1
                else:
                    argmin_pred = 2
                cum_sum[i + 1, j + 1] += candidates[argmin_pred]
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

@cython.boundscheck(False)
@cython.wraparound(False)
def dtw(numpy.ndarray[DTYPE_t, ndim=2] s1, numpy.ndarray[DTYPE_t, ndim=2] s2, numpy.ndarray[DTYPE_t, ndim=2] mask):
    assert s1.dtype == DTYPE and s2.dtype == DTYPE

    cdef int l1 = ts_size(s1)
    cdef int l2 = ts_size(s2)
    assert l1 <= mask.shape[0] and l2 <= mask.shape[1]

    cdef int i = 0
    cdef int j = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] cross_dist = cdist(s1[:l1], s2[:l2], "sqeuclidean").astype(DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] cum_sum = numpy.zeros((l1 + 1, l2 + 1), dtype=DTYPE)
    cum_sum[1:, 0] = numpy.inf
    cum_sum[0, 1:] = numpy.inf

    cross_dist[~numpy.isfinite(mask[:l1, :l2])] = numpy.inf

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = cross_dist[i, j]
            if numpy.isfinite(cum_sum[i + 1, j + 1]):
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j])

    return numpy.sqrt(cum_sum[l1, l2])

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cdist_dtw(numpy.ndarray[DTYPE_t, ndim=3] dataset1,
              numpy.ndarray[DTYPE_t, ndim=3] dataset2,
              numpy.ndarray[DTYPE_t, ndim=2] mask,
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
                cross_dist[i, j] = dtw(dataset1[i], dataset2[j], mask)

    return cross_dist





@cython.boundscheck(False)
@cython.wraparound(False)
def dtw_subsequence_path(numpy.ndarray[DTYPE_t, ndim=2] subseq, numpy.ndarray[DTYPE_t, ndim=2] longseq):
    assert subseq.dtype == DTYPE and longseq.dtype == DTYPE

    cdef int lsub = ts_size(subseq)
    cdef int llong = ts_size(longseq)

    cdef int i = 0
    cdef int j = 0
    cdef int argmin_pred = -1
    cdef numpy.ndarray[DTYPE_t, ndim=2] cross_dist = cdist(subseq[:lsub], longseq[:llong], "sqeuclidean").astype(DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] cum_sum = numpy.zeros((lsub + 1, llong + 1), dtype=DTYPE)
    cum_sum[1:, 0] = numpy.inf
    cdef numpy.ndarray[numpy.int_t, ndim=3] predecessors = numpy.zeros((lsub, llong, 2), dtype=numpy.int) - 1
    cdef numpy.ndarray[DTYPE_t, ndim=1] candidates = numpy.zeros((3, ), dtype=DTYPE)
    cdef list best_path

    for i in range(lsub):
        for j in range(llong):
            cum_sum[i + 1, j + 1] = cross_dist[i, j]
            candidates[0] = cum_sum[i, j + 1]
            candidates[1] = cum_sum[i + 1, j]
            candidates[2] = cum_sum[i, j]
            if candidates[0] <= candidates[1] and candidates[0] <= candidates[2]:
                argmin_pred = 0
            elif candidates[1] <= candidates[2]:
                argmin_pred = 1
            else:
                argmin_pred = 2
            cum_sum[i + 1, j + 1] += candidates[argmin_pred]
            if i > 0:
                if argmin_pred == 0:
                    predecessors[i, j, 0] = i - 1
                    predecessors[i, j, 1] = j
                elif argmin_pred == 1:
                    predecessors[i, j, 0] = i
                    predecessors[i, j, 1] = j - 1
                else:
                    predecessors[i, j, 0] = i - 1
                    predecessors[i, j, 1] = j - 1

    i = lsub - 1
    j = numpy.argmin(cum_sum[lsub, :]) - 1
    best_path = [(i, j)]
    while predecessors[i, j, 0] >= 0 and predecessors[i, j, 1] >= 0:
        i, j = predecessors[i, j, 0], predecessors[i, j, 1]
        best_path.insert(0, (i, j))

    return best_path, numpy.sqrt(cum_sum[lsub, best_path[len(best_path) - 1][1] + 1])


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def lb_envelope(numpy.ndarray[DTYPE_t, ndim=2] time_series, int radius):
    assert time_series.dtype == DTYPE
    cdef d = time_series.shape[1]
    cdef int sz = time_series.shape[0]
    cdef int i = 0
    cdef int min_idx = 0
    cdef int max_idx = 0
    cdef numpy.ndarray[DTYPE_t, ndim=2] enveloppe_up = numpy.empty((sz, d), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] enveloppe_down = numpy.empty((sz, d), dtype=DTYPE)

    for i in range(sz):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        enveloppe_down[i, :] = time_series[min_idx:max_idx, :].min(axis=0)
        enveloppe_up[i, :] = time_series[min_idx:max_idx, :].max(axis=0)

    return enveloppe_down, enveloppe_up


# @cython.boundscheck(False) # turn off bounds-checking for entire function
# @cython.wraparound(False)  # turn off negative index wrapping for entire function
# def cdist_lb_keogh(numpy.ndarray[DTYPE_t, ndim=3] dataset_queries, numpy.ndarray[DTYPE_t, ndim=3] dataset_candidates,
#                    int radius, bool self_similarity):
#     assert dataset_queries.dtype == DTYPE and dataset_candidates.dtype == DTYPE
#     cdef int n1 = dataset_queries.shape[0]
#     cdef int n2 = dataset_candidates.shape[0]
#     cdef int sz = dataset_candidates.shape[1]
#     cdef int i = 0
#     cdef int j = 0
#     cdef numpy.ndarray[DTYPE_t, ndim=2] cross_dist = numpy.empty((n1, n2), dtype=DTYPE)
#     cdef numpy.ndarray[DTYPE_t, ndim=2] env_u = numpy.empty((sz, 1), dtype=DTYPE)
#     cdef numpy.ndarray[DTYPE_t, ndim=2] env_d = numpy.empty((sz, 1), dtype=DTYPE)
#     cdef numpy.ndarray[DTYPE_t, ndim=3] enveloppes = numpy.empty((n2, sz, 2), dtype=DTYPE)
#
#     for j in range(n2):
#         env_d, env_u = lb_enveloppe(dataset_candidates, radius)
#         enveloppes[j, :, 0] = env_d
#         enveloppes[j, :, 1] = env_u
#
#     for i in range(n1):
#         for j in range(n2):
#             if self_similarity and j < i:
#                 cross_dist[i, j] = cross_dist[j, i]
#             elif self_similarity and i == j:
#                 cross_dist[i, j] = 0.
#             else:
#                 cross_dist[i, j] = lb_keogh(dataset_queries[i], (enveloppes[j, :, 0].reshape((-1, 1)),
#                                                                  enveloppes[j, :, 1].reshape((-1, 1))))
#
#     return cross_dist

