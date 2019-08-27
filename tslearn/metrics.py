"""
The :mod:`tslearn.metrics` module gathers time series similarity metrics.
"""

import numpy
from scipy.spatial.distance import pdist, cdist
from sklearn.utils import check_random_state
from tslearn.soft_dtw_fast import _soft_dtw, _soft_dtw_grad, \
    _jacobian_product_sq_euc
from sklearn.metrics.pairwise import euclidean_distances
from numba import njit, prange
from joblib import Parallel, delayed

from tslearn.cygak import cdist_normalized_gak as cycdist_normalized_gak, \
    normalized_gak as cynormalized_gak
from tslearn.utils import to_time_series, to_time_series_dataset, ts_size, \
    check_equal_size

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

GLOBAL_CONSTRAINT_CODE = {"": 0, "itakura": 1, "sakoe_chiba": 2}
VARIABLE_LENGTH_METRICS = ["dtw", "gak", "softdtw"]


@njit()
def _local_squared_dist(x, y):
    dist = 0.
    for di in prange(x.shape[0]):
        diff = (x[di] - y[di])
        dist += diff * diff
    return dist


@njit()
def njit_accumulated_matrix(s1, s2, mask):
    """Compute the accumulated cost matrix score between two time series.

    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.

    s2 : array, shape = (sz2,)
        Second time series

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        Accumulated cost matrix.

    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.

    for i in prange(l1):
        for j in prange(l2):
            if numpy.isfinite(mask[i, j]):
                cum_sum[i + 1, j + 1] = _local_squared_dist(s1[i], s2[j])
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                             cum_sum[i + 1, j],
                                             cum_sum[i, j])
    return cum_sum[1:, 1:]


@njit(nogil=True)
def njit_dtw(s1, s2, mask):
    """Compute the dynamic time warping score between two time series.

    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.

    s2 : array, shape = (sz2,)
        Second time series

    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    dtw_score : float
        Dynamic Time Warping score between both time series.

    """
    cum_sum = njit_accumulated_matrix(s1, s2, mask)
    return numpy.sqrt(cum_sum[-1, -1])


@njit()
def _return_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array([acc_cost_mat[i - 1][j - 1],
                               acc_cost_mat[i - 1][j],
                               acc_cost_mat[i][j - 1]])
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


@njit()
def njit_cdist_dtw(dataset1, dataset2, global_constraint, sakoe_chiba_radius,
                   itakura_max_slope):
    """Compute the cross-similarity matrix between two datasets.

    Parameters
    ----------
    dataset1 : array
        A dataset of time series.

    dataset2 : array
        Another dataset of time series.

    Returns
    -------
    cdist : array
        Cross-similarity matrix.
    """
    n_samples_1 = dataset1.shape[0]
    n_samples_2 = dataset2.shape[0]
    cdist_arr = numpy.empty((n_samples_1, n_samples_2))
    for i in prange(n_samples_1):
        s1 = dataset1[i]
        s1 = s1[numpy.isfinite(s1[:, 0])]
        for j in prange(n_samples_2):
            s2 = dataset2[j]
            s2 = s2[numpy.isfinite(s2[:, 0])]
            mask = compute_mask(s1, s2, global_constraint,
                                sakoe_chiba_radius, itakura_max_slope)
            cdist_arr[i, j] = njit_dtw(s1, s2, mask)
    return cdist_arr


@njit()
def njit_cdist_dtw_self(dataset, global_constraint, sakoe_chiba_radius,
                        itakura_max_slope):
    """Compute the cross-similarity matrix between a dataset and itself.

    Parameters
    ----------
    dataset : array
        A dataset of time series.

    Returns
    -------
    cdist : array
        Cross-similarity matrix.

    """
    n_samples = dataset.shape[0]
    cdist = numpy.zeros((n_samples, n_samples))
    for i in prange(n_samples):
        s1 = dataset[i]
        s1 = s1[numpy.isfinite(s1[:, 0])]
        for j in prange(i + 1, n_samples):
            s2 = dataset[j]
            s2 = s2[numpy.isfinite(s2[:, 0])]
            mask = compute_mask(s1, s2, global_constraint,
                                sakoe_chiba_radius, itakura_max_slope)
            cdist[i, j] = cdist[j, i] = njit_dtw(s1, s2, mask)
    return cdist


def dtw_path(s1, s2, global_constraint=None,
             sakoe_chiba_radius=1, itakura_max_slope=2.):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return both the path and the
    similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`P` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in P} (X_{i} - Y_{j})^2}

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.

    Parameters
    ----------
    s1
        A time series.
    s2
        Another time series.
        If not given, self-similarity of dataset1 is returned.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int (default: 1)
        Radius to be used for Sakoe-Chiba band global constraint. Used only if
        ``global_constraint="sakoe_chiba"``.
    itakura_max_slope : float (default: 2.)
        Maximum slope for the Itakura parallelogram constraint. Used only if
        ``global_constraint="itakura_parallelogram"``.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2
    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_path([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> dist
    0.0
    >>> dtw_path([1, 2, 3], [1., 2., 2., 3., 4.])[1]
    1.0

    See Also
    --------
    dtw : Get only the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    if global_constraint is not None:
        global_constraint_str = global_constraint
    else:
        global_constraint_str = ""

    mask = compute_mask(
        s1, s2, GLOBAL_CONSTRAINT_CODE[global_constraint_str],
        sakoe_chiba_radius, itakura_max_slope
    )
    acc_cost_mat = njit_accumulated_matrix(s1, s2, mask=mask)
    path = _return_path(acc_cost_mat)
    return path, numpy.sqrt(acc_cost_mat[-1, -1])


def dtw(s1, s2, global_constraint=None, sakoe_chiba_radius=1,
        itakura_max_slope=2.):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return it.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`P` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in P} (X_{i} - Y_{j})^2}

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.

    Parameters
    ----------
    s1
        A time series.
    s2
        Another time series.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int (default: 1)
        Radius to be used for Sakoe-Chiba band global constraint. Used only if
        ``global_constraint="sakoe_chiba"``.
    itakura_max_slope : float (default: 2.)
        Maximum slope for the Itakura parallelogram constraint. Used only if
        ``global_constraint="itakura"``.

    Returns
    -------
    float
        Similarity score

    Examples
    --------
    >>> dtw([1, 2, 3], [1., 2., 2., 3.])
    0.0
    >>> dtw([1, 2, 3], [1., 2., 2., 3., 4.])
    1.0

    See Also
    --------
    dtw_path : Get both the matching path and the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    if global_constraint is not None:
        global_constraint_str = global_constraint
    else:
        global_constraint_str = ""

    mask = compute_mask(
        s1, s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint_str],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope)
    return njit_dtw(s1, s2, mask=mask)


@njit()
def njit_accumulated_matrix_subsequence(subseq, longseq):
    """Compute the accumulated cost matrix score between a subsequence and
     a reference time series.

    Parameters
    ----------
    subseq : array, shape = (sz1, d)
        Subsequence time series.

    longseq : array, shape = (sz2, d)
        Reference time series

    Returns
    -------
    mat : array, shape = (sz1, sz2)
        Accumulated cost matrix.
    """
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, :] = 0.

    for i in prange(l1):
        for j in prange(l2):
            cum_sum[i + 1, j + 1] = _local_squared_dist(subseq[i], longseq[j])
            cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                         cum_sum[i + 1, j],
                                         cum_sum[i, j])
    return cum_sum[1:, 1:]


@njit()
def _return_path_subsequence(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    idx_last_match = numpy.argmin(acc_cost_mat[-1, :])
    path = [(sz1 - 1, idx_last_match)]
    while path[-1][0] != 0:
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array([acc_cost_mat[i - 1][j - 1],
                               acc_cost_mat[i - 1][j],
                               acc_cost_mat[i][j - 1]])
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def dtw_subsequence_path(subseq, longseq):
    r"""Compute sub-sequence Dynamic Time Warping (DTW) similarity measure
    between a (possibly multidimensional) query and a long time series and
    return both the path and the similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`P` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in P} (X_{i} - Y_{j})^2}

    Compared to traditional DTW, here, border constraints on admissible paths
    :math:`P` are relaxed such that :math:`P_0 = (0, ?)` and
    :math:`P_L = (N-1, ?)` where :math:`L` is the length of the considered path
    and :math:`N` is the length of the subsequence time series.

    It is not required that both time series share the same size, but they must
    be the same dimension. This implementation finds the best matching starting
    and ending positions for `subseq` inside `longseq`.

    Parameters
    ----------
    subseq
        A query time series.
    longseq
        A reference (supposed to be longer than `subseq`) time series.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`.
    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_subsequence_path([2., 3.], [1., 2., 2., 3., 4.])
    >>> path
    [(0, 2), (1, 3)]
    >>> dist
    0.0

    See Also
    --------
    dtw : Get the similarity score for DTW
    """
    subseq = to_time_series(subseq)
    longseq = to_time_series(longseq)
    acc_cost_mat = njit_accumulated_matrix_subsequence(subseq=subseq,
                                                       longseq=longseq)
    path = _return_path_subsequence(acc_cost_mat)
    return path, numpy.sqrt(numpy.min(acc_cost_mat[-1, :]))


@njit()
def sakoe_chiba_mask(sz1, sz2, radius=1):
    """Compute the Sakoe-Chiba mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    radius : int
        The radius of the band.

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Sakoe-Chiba mask.

    Examples
    --------
    >>> sakoe_chiba_mask(4, 4, 1)
    array([[ 0.,  0., inf, inf],
           [ 0.,  0.,  0., inf],
           [inf,  0.,  0.,  0.],
           [inf, inf,  0.,  0.]])
    >>> sakoe_chiba_mask(7, 3, 1)
    array([[ 0.,  0., inf],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  0.],
           [inf,  0.,  0.]])
    """
    mask = numpy.full((sz1, sz2), numpy.inf)
    if sz1 > sz2:
        width = sz1 - sz2 + radius
        for i in prange(sz2):
            lower = max(0, i - radius)
            upper = min(sz1, i + width) + 1
            mask[lower:upper, i] = 0.
    else:
        width = sz2 - sz1 + radius
        for i in prange(sz1):
            lower = max(0, i - radius)
            upper = min(sz2, i + width) + 1
            mask[i, lower:upper] = 0.
    return mask


@njit()
def itakura_mask(sz1, sz2, max_slope=2.):
    """Compute the Itakura mask.

    Parameters
    ----------
    sz1 : int
        The size of the first time series

    sz2 : int
        The size of the second time series.

    max_slope : float (default = 2)
        The maximum slope of the parallelogram.

    Returns
    -------
    mask : array, shape = (sz1, sz2)
        Itakura mask.

    Examples
    --------
    >>> itakura_mask(6, 6)
    array([[ 0., inf, inf, inf, inf, inf],
           [inf,  0.,  0., inf, inf, inf],
           [inf,  0.,  0.,  0., inf, inf],
           [inf, inf,  0.,  0.,  0., inf],
           [inf, inf, inf,  0.,  0., inf],
           [inf, inf, inf, inf, inf,  0.]])
    """
    min_slope = 1 / float(max_slope)
    max_slope *= (float(sz1) / float(sz2))
    min_slope *= (float(sz1) / float(sz2))

    lower_bound = numpy.empty((2, sz2))
    lower_bound[0] = min_slope * numpy.arange(sz2)
    lower_bound[1] = ((sz1 - 1) - max_slope * (sz2 - 1)
                      + max_slope * numpy.arange(sz2))
    lower_bound_ = numpy.empty(sz2)
    for i in prange(sz2):
        lower_bound_[i] = max(round(lower_bound[0, i], 2),
                              round(lower_bound[1, i], 2))
    lower_bound_ = numpy.ceil(lower_bound_)

    upper_bound = numpy.empty((2, sz2))
    upper_bound[0] = max_slope * numpy.arange(sz2)
    upper_bound[1] = ((sz1 - 1) - min_slope * (sz2 - 1)
                      + min_slope * numpy.arange(sz2))
    upper_bound_ = numpy.empty(sz2)
    for i in prange(sz2):
        upper_bound_[i] = min(round(upper_bound[0, i], 2),
                              round(upper_bound[1, i], 2))
    upper_bound_ = numpy.floor(upper_bound_ + 1)

    mask = numpy.full((sz1, sz2), numpy.inf)
    for i in prange(sz2):
        mask[int(lower_bound_[i]):int(upper_bound_[i]), i] = 0.

    return mask


@njit()
def compute_mask(s1, s2, global_constraint=0,
                 sakoe_chiba_radius=1, itakura_max_slope=2.):
    """Compute the mask (region constraint).

    Parameters
    ----------
    s1 : array
        A time series.
    s2: array
        Another time series.
    global_constraint : {0, 1, 2} (default: 0)
        Global constraint to restrict admissible paths for DTW:
        - "itakura" if 1
        - "sakoe_chiba" if 2
        - no constraint otherwise
    sakoe_chiba_radius : int (default: 1)
        Radius to be used for Sakoe-Chiba band global constraint. Used only if
        ``global_constraint="sakoe_chiba"``.
    itakura_max_slope : float (default: 2.)
        Maximum slope for the Itakura parallelogram constraint. Used only if
        ``global_constraint="itakura_parallelogram"``.

    Returns
    -------
    mask : array
        Constraint region.
    """
    sz1 = s1.shape[0]
    sz2 = s2.shape[0]
    if global_constraint == 1:
        mask = sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius)
    elif global_constraint == 2:
        mask = itakura_mask(sz1, sz2, max_slope=itakura_max_slope)
    else:
        mask = numpy.zeros((sz1, sz2))
    return mask


def cdist_dtw(dataset1, dataset2=None, global_constraint=None,
              sakoe_chiba_radius=1, itakura_max_slope=2., n_jobs=None):
    r"""Compute cross-similarity matrix using Dynamic Time Warping (DTW)
    similarity measure.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`P` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in P} (X_{i} - Y_{j})^2}

    DTW was originally presented in [1]_.

    Parameters
    ----------
    dataset1 : array-like
        A dataset of time series

    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

    sakoe_chiba_radius : int (default: 1)
        Radius to be used for Sakoe-Chiba band global constraint. Used only if
        ``global_constraint="sakoe_chiba"``.
    itakura_max_slope : float (default: 2.)
        Maximum slope for the Itakura parallelogram constraint. Used only if
        ``global_constraint="itakura"``.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]])
    array([[0., 1.],
           [1., 0.]])
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 3], [2, 3, 4, 5]])
    array([[0.        , 2.44948974],
           [1.        , 1.41421356]])

    See Also
    --------
    dtw : Get DTW similarity score

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    dataset1 = to_time_series_dataset(dataset1)

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = numpy.zeros((len(dataset1), len(dataset1)))
        indices = numpy.triu_indices(len(dataset1), k=1, m=len(dataset1))
        matrix[indices] = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(dtw)(
                dataset1[i], dataset1[j],
                global_constraint=global_constraint,
                sakoe_chiba_radius=sakoe_chiba_radius,
                itakura_max_slope=itakura_max_slope)
            for i in range(len(dataset1)) for j in range(i + 1, len(dataset1))
        )
        return matrix + matrix.T
    else:
        dataset2 = to_time_series_dataset(dataset2)
        matrix = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(dtw)(
                dataset1[i], dataset2[j],
                global_constraint=global_constraint,
                sakoe_chiba_radius=sakoe_chiba_radius,
                itakura_max_slope=itakura_max_slope)
            for i in range(len(dataset1)) for j in range(len(dataset2))
        )
        return numpy.array(matrix).reshape((len(dataset1), -1))


def cdist_dtw_no_parallel(dataset1, dataset2=None, global_constraint=None,
                          sakoe_chiba_radius=1, itakura_max_slope=2.):
    r"""Compute cross-similarity matrix using Dynamic Time Warping (DTW)
    similarity measure.
    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`P` is the alignment path:
    .. math::
        DTW(X, Y) = \sqrt{\sum_{(i, j) \in P} (X_{i} - Y_{j})^2}
    DTW was originally presented in [1]_.
    Parameters
    ----------
    dataset1 : array-like
        A dataset of time series
    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int (default: 1)
        Radius to be used for Sakoe-Chiba band global constraint. Used only if
        ``global_constraint="sakoe_chiba"``.
    itakura_max_slope : float (default: 2.)
        Maximum slope for the Itakura parallelogram constraint. Used only if
        ``global_constraint="itakura"``.

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_dtw_no_parallel([[1, 2, 2, 3], [1., 2., 3., 4.]])
    array([[0., 1.],
           [1., 0.]])
    >>> cdist_dtw_no_parallel([[1, 2, 2, 3], [1., 2., 3., 4.]],
    ...                       [[1, 2, 3], [2, 3, 4, 5]])
    array([[0.        , 2.44948974],
           [1.        , 1.41421356]])

    See Also
    --------
    dtw : Get DTW similarity score

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    dataset1 = to_time_series_dataset(dataset1)

    if global_constraint is not None:
        global_constraint_str = global_constraint
    else:
        global_constraint_str = ""

    if dataset2 is None:
        return njit_cdist_dtw_self(
            dataset1,
            GLOBAL_CONSTRAINT_CODE[global_constraint_str],
            sakoe_chiba_radius, itakura_max_slope)
    else:
        dataset2 = to_time_series_dataset(dataset2)
        return njit_cdist_dtw(dataset1, dataset2,
                              GLOBAL_CONSTRAINT_CODE[global_constraint_str],
                              sakoe_chiba_radius, itakura_max_slope)


@njit(nogil=True)
def njit_gak(s1, s2, gram):
    l1 = s1.shape[0]
    l2 = s2.shape[0]

    cum_sum = numpy.zeros((l1 + 1, l2 + 1))
    cum_sum[0, 0] = 1.

    for i in prange(l1):
        for j in prange(l2):
            cum_sum[i + 1, j + 1] = (cum_sum[i, j + 1] +
                                     cum_sum[i + 1, j] +
                                     cum_sum[i, j]) * gram[i, j]

    return cum_sum[l1, l2]


def _gak_gram(s1, s2, sigma=1.):
    gram = - cdist(s1, s2, "sqeuclidean") / (2 * sigma ** 2)
    gram -= numpy.log(2 - numpy.exp(gram))
    return numpy.exp(gram)


def unnormalized_gak(s1, s2, sigma=1.):
    r"""Compute Global Alignment Kernel (GAK) between (possibly
    multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must
    be the same dimension. GAK was
    originally presented in [1]_.
    This is an unnormalized version.

    Parameters
    ----------
    s1
        A time series
    s2
        Another time series
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK

    Returns
    -------
    float
        Kernel value

    Examples
    --------
    >>> unnormalized_gak([1, 2, 3],
    ...                  [1., 2., 2., 3.],
    ...                  sigma=2.)  # doctest: +ELLIPSIS
    15.358...
    >>> unnormalized_gak([1, 2, 3],
    ...                  [1., 2., 2., 3., 4.])  # doctest: +ELLIPSIS
    3.166...

    See Also
    --------
    gak : normalized version of GAK that ensures that k(x,x) = 1
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    gram = _gak_gram(s1, s2, sigma=sigma)

    gak_val = njit_gak(s1, s2, gram)
    return gak_val

def gak(s1, s2, sigma=1.):  # TODO: better doc (formula for the kernel)
    r"""Compute Global Alignment Kernel (GAK) between (possibly
    multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must
    be the same dimension. GAK was
    originally presented in [1]_.
    This is a normalized version that ensures that :math:`k(x,x)=1` for all
    :math:`x` and :math:`k(x,y) \in [0, 1]` for all :math:`x, y`.

    Parameters
    ----------
    s1
        A time series
    s2
        Another time series
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK

    Returns
    -------
    float
        Kernel value

    Examples
    --------
    >>> gak([1, 2, 3], [1., 2., 2., 3.], sigma=2.)  # doctest: +ELLIPSIS
    0.839...
    >>> gak([1, 2, 3], [1., 2., 2., 3., 4.])  # doctest: +ELLIPSIS
    0.273...

    See Also
    --------
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    denom = numpy.sqrt(unnormalized_gak(s1, s1, sigma=sigma) *
                       unnormalized_gak(s2, s2, sigma=sigma))
    return unnormalized_gak(s1, s2, sigma=sigma) / denom


def cdist_gak(dataset1, dataset2=None, sigma=1., n_jobs=None):
    r"""Compute cross-similarity matrix using Global Alignment kernel (GAK).

    GAK was originally presented in [1]_.

    Parameters
    ----------
    dataset1
        A dataset of time series
    dataset2
        Another dataset of time series
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_gak([[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)
    array([[1.        , 0.65629661],
           [0.65629661, 1.        ]])
    >>> cdist_gak([[1, 2, 2], [1., 2., 3., 4.]],
    ...           [[1, 2, 2, 3], [1., 2., 3., 4.], [1, 2, 2, 3]],
    ...           sigma=2.)
    array([[0.71059484, 0.29722877, 0.71059484],
           [0.65629661, 1.        , 0.65629661]])

    See Also
    --------
    gak : Compute Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    dataset1 = to_time_series_dataset(dataset1)

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = numpy.zeros((len(dataset1), len(dataset1)))
        indices = numpy.triu_indices(len(dataset1), k=0, m=len(dataset1))
        matrix[indices] = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(unnormalized_gak)(dataset1[i], dataset1[j], sigma=sigma)
            for i in range(len(dataset1)) for j in range(i, len(dataset1))
        )
        indices = numpy.tril_indices(len(dataset1), k=-1, m=len(dataset1))
        matrix[indices] = matrix.T[indices]
        diagonal = numpy.diag(numpy.sqrt(1. / numpy.diag(matrix)))
        diagonal_left = diagonal_right = diagonal
    else:
        dataset2 = to_time_series_dataset(dataset2)
        matrix = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(unnormalized_gak)(dataset1[i], dataset2[j], sigma=sigma)
            for i in range(len(dataset1)) for j in range(len(dataset2))
        )
        matrix = numpy.array(matrix).reshape((len(dataset1), -1))
        diagonal_left = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(unnormalized_gak)(dataset1[i], dataset1[i], sigma=sigma)
            for i in range(len(dataset1))
        )
        diagonal_right = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(unnormalized_gak)(dataset2[j], dataset2[j], sigma=sigma)
            for j in range(len(dataset2))
        )
        diagonal_left = numpy.diag(1. / numpy.sqrt(diagonal_left))
        diagonal_right = numpy.diag(1. / numpy.sqrt(diagonal_right))
    return (diagonal_left.dot(matrix)).dot(diagonal_right)


def cdist_gak_no_parallel(dataset1, dataset2=None, sigma=1.):
    r"""Compute cross-similarity matrix using Global Alignment kernel (GAK).

    GAK was originally presented in [1]_.

    Parameters
    ----------
    dataset1
        A dataset of time series
    dataset2
        Another dataset of time series
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_gak_no_parallel([[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)
    array([[1.        , 0.65629661],
           [0.65629661, 1.        ]])
    >>> cdist_gak_no_parallel([[1, 2, 2], [1., 2., 3., 4.]],
    ...           [[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)
    array([[0.71059484, 0.29722877],
           [0.65629661, 1.        ]])

    See Also
    --------
    gak : Compute Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    dataset1 = to_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = to_time_series_dataset(dataset2)
    return cycdist_normalized_gak(dataset1, dataset2, sigma,
                                  self_similarity=self_similarity)


def sigma_gak(dataset, n_samples=100, random_state=None):
    r"""Compute sigma value to be used for GAK.

    This method was originally presented in [1]_.

    Parameters
    ----------
    dataset
        A dataset of time series
    n_samples : int (default: 100)
        Number of samples on which median distance should be estimated
    random_state : integer or numpy.RandomState or None (default: None)
        The generator used to draw the samples. If an integer is given, it
        fixes the seed. Defaults to the global numpy random number generator.

    Returns
    -------
    float
        Suggested bandwidth (:math:`\\sigma`) for the Global Alignment kernel

    Examples
    --------
    >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
    >>> sigma_gak(dataset=dataset,
    ...           n_samples=200,
    ...           random_state=0)  # doctest: +ELLIPSIS
    2.0...

    See Also
    --------
    gak : Compute Global Alignment kernel
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    random_state = check_random_state(random_state)
    dataset = to_time_series_dataset(dataset)
    n_ts, sz, d = dataset.shape
    if not check_equal_size(dataset):
        sz = numpy.min([ts_size(ts) for ts in dataset])
    if n_ts * sz < n_samples:
        replace = True
    else:
        replace = False
    sample_indices = random_state.choice(n_ts * sz,
                                         size=n_samples,
                                         replace=replace)
    dists = pdist(dataset[:, :sz, :].reshape((-1, d))[sample_indices],
                  metric="euclidean")
    return numpy.median(dists) * numpy.sqrt(sz)


def gamma_soft_dtw(dataset, n_samples=100, random_state=None):
    r"""Compute gamma value to be used for GAK/Soft-DTW.

    This method was originally presented in [1]_.

    Parameters
    ----------
    dataset
        A dataset of time series
    n_samples : int (default: 100)
        Number of samples on which median distance should be estimated
    random_state : integer or numpy.RandomState or None (default: None)
        The generator used to draw the samples. If an integer is given, it
        fixes the seed. Defaults to the global numpy random number generator.

    Returns
    -------
    float
        Suggested :math:`\\gamma` parameter for the Soft-DTW

    Examples
    --------
    >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
    >>> gamma_soft_dtw(dataset=dataset,
    ...                n_samples=200,
    ...                random_state=0)  # doctest: +ELLIPSIS
    8.0...

    See Also
    --------
    sigma_gak : Compute sigma parameter for Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    return 2. * sigma_gak(dataset=dataset,
                          n_samples=n_samples,
                          random_state=random_state) ** 2


def lb_keogh(ts_query, ts_candidate=None, radius=1, envelope_candidate=None):
    r"""Compute LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts_query : array-like
        Query time-series to compare to the envelope of the candidate.
    ts_candidate : array-like or None (default: None)
        Candidate time-series. None means the envelope is provided via
        `envelope_query` parameter and hence does not
        need to be computed again.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on
        all observations from the candidate time series at indices comprised
        between i-radius and i+radius). Not used
        if `ts_candidate` is None.
    envelope_candidate: pair of array-like (envelope_down, envelope_up) or None
    (default: None)
        Pre-computed envelope of the candidate time series. If set to None, it
        is computed based on `ts_candidate`.

    Notes
    -----
        This method requires a `ts_query` and `ts_candidate` (or
        `envelope_candidate`, depending on the call) to be of equal size.

    Returns
    -------
    float
        Distance between the query time series and the envelope of the
        candidate time series.

    Examples
    --------
    >>> ts1 = [1, 2, 3, 2, 1]
    >>> ts2 = [0, 0, 0, 0, 0]
    >>> env_low, env_up = lb_envelope(ts1, radius=1)
    >>> lb_keogh(ts_query=ts2,
    ...          envelope_candidate=(env_low, env_up))  # doctest: +ELLIPSIS
    2.8284...
    >>> lb_keogh(ts_query=ts2,
    ...          ts_candidate=ts1,
    ...          radius=1)  # doctest: +ELLIPSIS
    2.8284...

    See also
    --------
    lb_envelope : Compute LB_Keogh-related envelope

    References
    ----------
    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """
    if ts_candidate is None:
        envelope_down, envelope_up = envelope_candidate
    else:
        ts_candidate = to_time_series(ts_candidate)
        assert ts_candidate.shape[1] == 1, \
            "LB_Keogh is available only for monodimensional time series"
        envelope_down, envelope_up = lb_envelope(ts_candidate, radius)
    ts_query = to_time_series(ts_query)
    assert ts_query.shape[1] == 1, \
        "LB_Keogh is available only for monodimensional time series"
    indices_up = ts_query[:, 0] > envelope_up[:, 0]
    indices_down = ts_query[:, 0] < envelope_down[:, 0]
    return numpy.sqrt(numpy.linalg.norm(ts_query[indices_up, 0] -
                                        envelope_up[indices_up, 0]) ** 2 +
                      numpy.linalg.norm(ts_query[indices_down, 0] -
                                        envelope_down[indices_down, 0]) ** 2)


@njit()
def njit_lb_envelope(time_series, radius):
    sz, d = time_series.shape
    enveloppe_up = numpy.empty((sz, d))
    enveloppe_down = numpy.empty((sz, d))

    for i in prange(sz):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        for di in prange(d):
            enveloppe_down[i, di] = numpy.min(time_series[min_idx:max_idx, di])
            enveloppe_up[i, di] = numpy.max(time_series[min_idx:max_idx, di])

    return enveloppe_down, enveloppe_up


def lb_envelope(ts, radius=1):
    r"""Compute time-series envelope as required by LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts : array-like
        Time-series for which the envelope should be computed.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on
        all observations from the time series at indices comprised between
        i-radius and i+radius).

    Returns
    -------
    array-like
        Lower-side of the envelope.
    array-like
        Upper-side of the envelope.

    Examples
    --------
    >>> ts1 = [1, 2, 3, 2, 1]
    >>> env_low, env_up = lb_envelope(ts1, radius=1)
    >>> env_low
    array([[1.],
           [1.],
           [2.],
           [1.],
           [1.]])
    >>> env_up
    array([[2.],
           [3.],
           [3.],
           [3.],
           [2.]])

    See also
    --------
    lb_keogh : Compute LB_Keogh similarity

    References
    ----------
    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """
    return njit_lb_envelope(to_time_series(ts), radius=radius)


def soft_dtw(ts1, ts2, gamma=1.):
    r"""Compute Soft-DTW metric between two time series.

    Soft-DTW was originally presented in [1]_.

    Parameters
    ----------
    ts1
        A time series
    ts2
        Another time series
    gamma : float (default 1.)
        Gamma paraneter for Soft-DTW

    Returns
    -------
    float
        Similarity

    Examples
    --------
    >>> soft_dtw([1, 2, 2, 3],
    ...          [1., 2., 3., 4.],
    ...          gamma=1.)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    -0.89...
    >>> soft_dtw([1, 2, 3, 3],
    ...          [1., 2., 2.1, 3.2],
    ...          gamma=0.01)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    0.089...

    See Also
    --------
    cdist_soft_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    if gamma == 0.:
        return dtw(ts1, ts2)
    return SoftDTW(SquaredEuclidean(ts1[:ts_size(ts1)], ts2[:ts_size(ts2)]),
                   gamma=gamma).compute()


def cdist_soft_dtw(dataset1, dataset2=None, gamma=1.):
    r"""Compute cross-similarity matrix using Soft-DTW metric.

    Soft-DTW was originally presented in [1]_.

    Parameters
    ----------
    dataset1
        A dataset of time series
    dataset2
        Another dataset of time series
    gamma : float (default 1.)
        Gamma paraneter for Soft-DTW

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], gamma=.01)
    array([[-0.01098612,  1.        ],
           [ 1.        ,  0.        ]])
    >>> cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]],
    ...                [[1, 2, 2, 3], [1., 2., 3., 4.]], gamma=.01)
    array([[-0.01098612,  1.        ],
           [ 1.        ,  0.        ]])

    See Also
    --------
    soft_dtw : Compute Soft-DTW
    cdist_soft_dtw_normalized : Cross similarity matrix between time series
        datasets using a normalized version of Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    dataset1 = to_time_series_dataset(dataset1, dtype=numpy.float64)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = to_time_series_dataset(dataset2, dtype=numpy.float64)
    dists = numpy.empty((dataset1.shape[0], dataset2.shape[0]))
    equal_size_ds1 = check_equal_size(dataset1)
    equal_size_ds2 = check_equal_size(dataset2)
    for i, ts1 in enumerate(dataset1):
        if equal_size_ds1:
            ts1_short = ts1
        else:
            ts1_short = ts1[:ts_size(ts1)]
        for j, ts2 in enumerate(dataset2):
            if equal_size_ds2:
                ts2_short = ts2
            else:
                ts2_short = ts2[:ts_size(ts2)]
            if self_similarity and j < i:
                dists[i, j] = dists[j, i]
            else:
                dists[i, j] = soft_dtw(ts1_short, ts2_short, gamma=gamma)

    return dists


def cdist_soft_dtw_normalized(dataset1, dataset2=None, gamma=1.):
    """Compute cross-similarity matrix using a normalized version of the
    Soft-DTW metric.

    Soft-DTW was originally presented in [1]_.
    This normalized version is defined as:
    `sdtw_(x,y) := sdtw(x,y) - 1/2(sdtw(x,x)+sdtw(y,y))`
    and ensures that all returned values are positive and that
    `sdtw_(x,x) == 0`.

    Parameters
    ----------
    dataset1
        A dataset of time series
    dataset2
        Another dataset of time series
    gamma : float (default 1.)
        Gamma paraneter for Soft-DTW

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> time_series = numpy.random.randn(10, 15, 1)
    >>> numpy.alltrue(cdist_soft_dtw_normalized(time_series) >= 0.)
    True

    See Also
    --------
    soft_dtw : Compute Soft-DTW
    cdist_soft_dtw : Cross similarity matrix between time series
        datasets using the unnormalized version of Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    dists = cdist_soft_dtw(dataset1, dataset2=dataset2, gamma=gamma)
    d_ii = numpy.diag(dists)
    dists -= .5 * (d_ii.reshape((-1, 1)) + d_ii.reshape((1, -1)))
    return dists


class SoftDTW(object):
    def __init__(self, D, gamma=1.):
        """
        Parameters
        ----------
        gamma: float
            Regularization parameter.
            Lower is less smoothed (closer to true DTW).

        Attributes
        ----------
        self.R_: array, shape = [m + 2, n + 2]
            Accumulated cost matrix (stored after calling `compute`).
        """
        if hasattr(D, "compute"):
            self.D = D.compute()
        else:
            self.D = D
        self.D = self.D.astype(numpy.float64)

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the backward recursion.
        m, n = self.D.shape
        self.R_ = numpy.zeros((m + 2, n + 2), dtype=numpy.float64)
        self.computed = False

        self.gamma = numpy.float64(gamma)

    def compute(self):
        """Compute soft-DTW by dynamic programming.

        Returns
        -------
        sdtw: float
            soft-DTW discrepancy.
        """
        m, n = self.D.shape

        _soft_dtw(self.D, self.R_, gamma=self.gamma)

        self.computed = True

        return self.R_[m, n]

    def grad(self):
        """Compute gradient of soft-DTW w.r.t. D by dynamic programming.

        Returns
        -------
        grad: array, shape = [m, n]
            Gradient w.r.t. D.
        """
        if not self.computed:
            raise ValueError("Needs to call compute() first.")

        m, n = self.D.shape

        # Add an extra row and an extra column to D.
        # Needed to deal with edge cases in the recursion.
        D = numpy.vstack((self.D, numpy.zeros(n)))
        D = numpy.hstack((D, numpy.zeros((m + 1, 1))))

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the recursion.
        E = numpy.zeros((m + 2, n + 2), dtype=numpy.float64)

        _soft_dtw_grad(D, self.R_, E, gamma=self.gamma)

        return E[1:-1, 1:-1]


class SquaredEuclidean(object):
    def __init__(self, X, Y):
        """
        Parameters
        ----------
        X: array, shape = [m, d]
            First time series.
        Y: array, shape = [n, d]
            Second time series.

        Examples
        --------
        >>> SquaredEuclidean([1, 2, 2, 3], [1, 2, 3, 4]).compute()
        array([[0., 1., 4., 9.],
               [1., 0., 1., 4.],
               [1., 0., 1., 4.],
               [4., 1., 0., 1.]])
        """
        self.X = to_time_series(X).astype(numpy.float64)
        self.Y = to_time_series(Y).astype(numpy.float64)

    def compute(self):
        """Compute distance matrix.

        Returns
        -------
        D: array, shape = [m, n]
            Distance matrix.
        """
        return euclidean_distances(self.X, self.Y, squared=True)

    def jacobian_product(self, E):
        """Compute the product between the Jacobian
        (a linear map from m x d to m x n) and a matrix E.

        Parameters
        ----------
        E: array, shape = [m, n]
            Second time series.

        Returns
        -------
        G: array, shape = [m, d]
            Product with Jacobian
            ([m x d, m x n] * [m x n] = [m x d]).
        """
        G = numpy.zeros_like(self.X, dtype=numpy.float64)

        _jacobian_product_sq_euc(self.X, self.Y, E.astype(numpy.float64), G)

        return G
