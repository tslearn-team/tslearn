import numpy

from numba import njit, prange

from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series

from ._masks import compute_mask as compute_mask_

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}


@njit()
def _njit_local_squared_dist(x, y):
    """Compute the squared distance between two vectors.

    Parameters
    ----------
    x : array-like, shape=(d,)
        A vector.
    y : array-like, shape=(d,)
        Another vector.

    Returns
    -------
    dist : float
        Squared distance between x and y.
    """
    dist = 0.0
    for di in range(x.shape[0]):
        diff = x[di] - y[di]
        dist += diff * diff
    return dist


def _local_squared_dist(x, y, be=None):
    """Compute the squared distance between two vectors.

    Parameters
    ----------
    x : array-like, shape=(d,)
        A vector.
    y : array-like, shape=(d,)
        Another vector.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    dist : float
        Squared distance between x and y.
    """
    be = instantiate_backend(be, x, y)
    x = be.array(x)
    y = be.array(y)
    dist = 0.0
    for di in range(be.shape(x)[0]):
        diff = x[di] - y[di]
        dist += diff * diff
    return dist


def _max_steps(i, j, max_length, length_1, length_2):
    """Maximum number of steps required in a L-DTW process to reach a given
    cell.

    Parameters
    ----------
    i : int
        Cell row index

    j : int
        Cell column index

    max_length : int
        Maximum allowed length

    length_1 : int
        Length of the first time series

    length_2 : int
        Length of the second time series

    Returns
    -------
    int
        Number of steps
    """
    candidate_1 = i + j
    candidate_2 = max_length - max(length_1 - i - 1, length_2 - j - 1)
    return min(candidate_1, candidate_2)


def _limited_warping_length_cost(s1, s2, max_length, be=None):
    r"""Compute accumulated scores necessary fo L-DTW.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        A time series.
    s2 : array-like, shape=(sz2, d)
        Another time series.
    max_length : int
        Maximum allowed warping path length. Should be an integer between
        XXX and YYY.  # TODO
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    dict_costs : dict
        Accumulated scores. This dict associates (i, j) pairs (keys) to
        dictionaries with desired length as key and associated score as value.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = be.array(s1)
    s2 = be.array(s2)
    dict_costs = {}
    for i in range(s1.shape[0]):
        for j in range(s2.shape[0]):
            dict_costs[i, j] = {}

    # Init
    dict_costs[0, 0][0] = _local_squared_dist(s1[0], s2[0], be=be)
    for i in range(1, s1.shape[0]):
        pred = dict_costs[i - 1, 0][i - 1]
        dict_costs[i, 0][i] = pred + _local_squared_dist(s1[i], s2[0], be=be)
    for j in range(1, s2.shape[0]):
        pred = dict_costs[0, j - 1][j - 1]
        dict_costs[0, j][j] = pred + _local_squared_dist(s1[0], s2[j], be=be)

    # Main loop
    for i in range(1, s1.shape[0]):
        for j in range(1, s2.shape[0]):
            min_s = max(i, j)
            max_s = _max_steps(i, j, max_length - 1, s1.shape[0], s2.shape[0])
            for s in range(min_s, max_s + 1):
                dict_costs[i, j][s] = _local_squared_dist(s1[i], s2[j], be=be)
                dict_costs[i, j][s] += min(
                    dict_costs[i, j - 1].get(s - 1, be.inf),
                    dict_costs[i - 1, j].get(s - 1, be.inf),
                    dict_costs[i - 1, j - 1].get(s - 1, be.inf),
                )
    return dict_costs


def dtw_limited_warping_length(s1, s2, max_length, be=None):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series under an upper bound constraint on
    the resulting path length and return the similarity cost.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the optimal alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.
    This constrained-length variant was introduced in [2]_.
    Both bariants are
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    max_length : int
        Maximum allowed warping path length.
        If greater than len(s1) + len(s2), then it is equivalent to
        unconstrained DTW.
        If lower than max(len(s1), len(s2)), no path can be found and a
        ValueError is raised.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    float
        Similarity score

    Examples
    --------
    >>> float(dtw_limited_warping_length([1, 2, 3], [1., 2., 2., 3.], 5))
    0.0
    >>> float(dtw_limited_warping_length([1, 2, 3], [1., 2., 2., 3., 4.], 5))
    1.0

    See Also
    --------
    dtw : Get the similarity score for DTW
    dtw_path_limited_warping_length : Get both the warping path and the
        similarity score for DTW with limited warping path length

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    .. [2] Z. Zhang, R. Tavenard, A. Bailly, X. Tang, P. Tang, T. Corpetti
           Dynamic time warping under limited warping path length.
           Information Sciences, vol. 393, pp. 91--107, 2017.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if max_length < max(s1.shape[0], s2.shape[0]):
        raise ValueError(
            "Cannot find a path of length {} to align given "
            "time series.".format(max_length)
        )

    accumulated_costs = _limited_warping_length_cost(s1, s2, max_length, be=be)
    idx_pair = (s1.shape[0] - 1, s2.shape[0] - 1)
    optimal_cost = min(accumulated_costs[idx_pair].values())
    return be.sqrt(optimal_cost)


def _return_path_limited_warping_length(
    accum_costs, target_indices, optimal_length, be=None
):
    """Return the optimal path under an upper bound constraint on
    the path length.

    Parameters
    ----------
    accum_costs : dict
        Accumulated scores. This dict associates (i, j) pairs (keys) to
        dictionaries with desired length as key and associated score as value.
    target_indices : tuple (a pair of integers)
        Target indices.
    optimal_length : int
        Optimal length.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    path : list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to a first time series s1 and the second one
        corresponds to a second time series s2.
    """
    be = instantiate_backend(be)
    path = [target_indices]
    cur_length = optimal_length
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = be.array(
                [
                    accum_costs[i - 1, j - 1].get(cur_length - 1, be.inf),
                    accum_costs[i - 1, j].get(cur_length - 1, be.inf),
                    accum_costs[i, j - 1].get(cur_length - 1, be.inf),
                ]
            )
            argmin = be.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
            cur_length -= 1
    return path[::-1]


def dtw_path_limited_warping_length(s1, s2, max_length, be=None):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series under an upper bound constraint on
    the resulting path length and return the path as well as the similarity
    cost.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the optimal alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.
    This constrained-length variant was introduced in [2]_.
    Both variants are
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    max_length : int
        Maximum allowed warping path length.
        If greater than len(s1) + len(s2), then it is equivalent to
        unconstrained DTW.
        If lower than max(len(s1), len(s2)), no path can be found and a
        ValueError is raised.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    list of integer pairs
        Optimal path
    float
        Similarity score

    Examples
    --------
    >>> path, cost = dtw_path_limited_warping_length([1, 2, 3],
    ...                                              [1., 2., 2., 3.], 5)
    >>> float(cost)
    0.0
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> path, cost = dtw_path_limited_warping_length([1, 2, 3],
    ...                                              [1., 2., 2., 3., 4.], 5)
    >>> float(cost)
    1.0
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4)]

    See Also
    --------
    dtw_limited_warping_length : Get the similarity score for DTW with limited
        warping path length
    dtw_path : Get both the matching path and the similarity score for DTW

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    .. [2] Z. Zhang, R. Tavenard, A. Bailly, X. Tang, P. Tang, T. Corpetti
           Dynamic time warping under limited warping path length.
           Information Sciences, vol. 393, pp. 91--107, 2017.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if max_length < max(s1.shape[0], s2.shape[0]):
        raise ValueError(
            "Cannot find a path of length {} to align given "
            "time series.".format(max_length)
        )

    accumulated_costs = _limited_warping_length_cost(s1, s2, max_length, be=be)
    idx_pair = (s1.shape[0] - 1, s2.shape[0] - 1)
    optimal_length = -1
    optimal_cost = be.inf
    for k, v in accumulated_costs[idx_pair].items():
        if v < optimal_cost:
            optimal_cost = v
            optimal_length = k
    path = _return_path_limited_warping_length(
        accumulated_costs, idx_pair, optimal_length, be=be
    )
    return path, be.sqrt(optimal_cost)


@njit()
def _njit_subsequence_cost_matrix(subseq, longseq):
    """Compute the accumulated cost matrix score between a subsequence and
    a reference time series.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d)
        Subsequence time series.
    longseq : array-like, shape=(sz2, d)
        Reference time series.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, :] = 0.0

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = _njit_local_squared_dist(subseq[i], longseq[j])
            cum_sum[i + 1, j + 1] += min(
                cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]
            )
    return cum_sum[1:, 1:]


def _subsequence_cost_matrix(subseq, longseq, be=None):
    """Compute the accumulated cost matrix score between a subsequence and
    a reference time series.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d)
        Subsequence time series.
    longseq : array-like, shape=(sz2, d)
        Reference time series.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    be = instantiate_backend(be, subseq, longseq)
    subseq = be.array(subseq)
    longseq = be.array(longseq)
    l1 = subseq.shape[0]
    l2 = longseq.shape[0]
    cum_sum = be.full((l1 + 1, l2 + 1), be.inf)
    cum_sum[0, :] = 0.0

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = _local_squared_dist(subseq[i], longseq[j], be=be)
            cum_sum[i + 1, j + 1] += min(
                cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j]
            )
    return cum_sum[1:, 1:]


def subsequence_cost_matrix(subseq, longseq, be=None):
    """Compute the accumulated cost matrix score between a subsequence and
    a reference time series.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d) or (sz1,)
        Subsequence time series. If shape is (sz1,), the time series is assumed to be univariate.
    longseq : array-like, shape=(sz2, d) or (sz2,)
        Reference time series. If shape is (sz2,), the time series is assumed to be univariate.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    be = instantiate_backend(be, subseq, longseq)
    subseq = to_time_series(subseq, remove_nans=True, be=be)
    longseq = to_time_series(longseq, remove_nans=True, be=be)
    if be.is_numpy:
        return _njit_subsequence_cost_matrix(subseq, longseq)
    return _subsequence_cost_matrix(subseq, longseq, be=be)


@njit()
def _njit_subsequence_path(acc_cost_mat, idx_path_end):
    r"""Compute the optimal path through an accumulated cost matrix given the
    endpoint of the sequence.

    Parameters
    ----------
    acc_cost_mat: array-like, shape=(sz1, sz2)
        Accumulated cost matrix comparing subsequence from a longer sequence.
    idx_path_end: int
        The end position of the matched subsequence in the longer sequence.

    Returns
    -------
    path: list of tuples of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`. The startpoint of the Path is :math:`P_0 = (0, ?)` and it
        ends at :math:`P_L = (len(subseq)-1, idx\_path\_end)`
    """
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, idx_path_end)]
    while path[-1][0] != 0:
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = numpy.array(
                [
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1],
                ]
            )
            argmin = numpy.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def _subsequence_path(acc_cost_mat, idx_path_end, be=None):
    r"""Compute the optimal path through an accumulated cost matrix given the
    endpoint of the sequence.

    Parameters
    ----------
    acc_cost_mat: array-like, shape=(sz1, sz2)
        Accumulated cost matrix comparing subsequence from a longer sequence.
    idx_path_end: int
        The end position of the matched subsequence in the longer sequence.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    path: list of tuples of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`. The startpoint of the Path is :math:`P_0 = (0, ?)` and it
        ends at :math:`P_L = (len(subseq)-1, idx\_path\_end)`
    """
    be = instantiate_backend(be, acc_cost_mat)
    acc_cost_mat = be.array(acc_cost_mat)
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, idx_path_end)]
    while path[-1][0] != 0:
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = be.array(
                [
                    acc_cost_mat[i - 1][j - 1],
                    acc_cost_mat[i - 1][j],
                    acc_cost_mat[i][j - 1],
                ]
            )
            argmin = be.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


def subsequence_path(acc_cost_mat, idx_path_end, be=None):
    r"""Compute the optimal path through an accumulated cost matrix given the
    endpoint of the sequence.

    Parameters
    ----------
    acc_cost_mat: array-like, shape=(sz1, sz2)
        Accumulated cost matrix comparing subsequence from a longer sequence.
    idx_path_end: int
        The end position of the matched subsequence in the longer sequence.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    path: list of tuples of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to `subseq` and the second one corresponds to
        `longseq`. The startpoint of the Path is :math:`P_0 = (0, ?)` and it
        ends at :math:`P_L = (len(subseq)-1, idx\_path\_end)`

    Examples
    --------

    >>> acc_cost_mat = numpy.array([[1., 0., 0., 1., 4.],
    ...                             [5., 1., 1., 0., 1.]])
    >>> # calculate the globally optimal path
    >>> optimal_end_point = numpy.argmin(acc_cost_mat[-1, :])
    >>> path = subsequence_path(acc_cost_mat, optimal_end_point)
    >>> path
    [(0, 2), (1, 3)]

    See Also
    --------
    dtw_subsequence_path : Get the similarity score for DTW
    subsequence_cost_matrix: Calculate the required cost matrix

    """
    be = instantiate_backend(be, acc_cost_mat)
    acc_cost_mat = be.array(acc_cost_mat)
    if be.is_numpy:
        return _njit_subsequence_path(acc_cost_mat, idx_path_end)
    return _subsequence_path(acc_cost_mat, idx_path_end, be=be)


def dtw_subsequence_path(subseq, longseq, be=None):
    r"""Compute sub-sequence Dynamic Time Warping (DTW) similarity measure
    between a (possibly multidimensional) query and a long time series and
    return both the path and the similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Compared to traditional DTW, here, border constraints on admissible paths
    :math:`\pi` are relaxed such that :math:`\pi_0 = (0, ?)` and
    :math:`\pi_L = (N-1, ?)` where :math:`L` is the length of the considered
    path and :math:`N` is the length of the subsequence time series.

    It is not required that both time series share the same size, but they must
    be the same dimension. This implementation finds the best matching starting
    and ending positions for `subseq` inside `longseq`.

    Parameters
    ----------
    subseq : array-like, shape=(sz1, d) or (sz1,)
        A query time series.
        If shape is (sz1,), the time series is assumed to be univariate.
    longseq : array-like, shape=(sz2, d) or (sz2,)
        A reference (supposed to be longer than `subseq`) time series.
        If shape is (sz2,), the time series is assumed to be univariate.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

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
    >>> float(dist)
    0.0

    See Also
    --------
    dtw : Get the similarity score for DTW
    subsequence_cost_matrix: Calculate the required cost matrix
    subsequence_path: Calculate a matching path manually
    """
    be = instantiate_backend(be, subseq, longseq)
    subseq = to_time_series(subseq, be=be)
    longseq = to_time_series(longseq, be=be)
    acc_cost_mat = subsequence_cost_matrix(subseq=subseq, longseq=longseq, be=be)
    global_optimal_match = be.argmin(acc_cost_mat[-1, :])
    path = subsequence_path(acc_cost_mat, global_optimal_match, be=be)
    return path, be.sqrt(acc_cost_mat[-1, :][global_optimal_match])


def lb_keogh(ts_query, ts_candidate=None, radius=1, envelope_candidate=None):
    r"""Compute LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts_query : array-like, shape=(sz1, 1) or (sz1,)
        Univariate query time series to compare to the envelope of the candidate.
    ts_candidate : None or array-like, shape=(sz2, 1) or (sz2,) (default: None)
        Univariate candidate time series. None means the envelope is provided via
        `envelope_candidate` parameter and hence does not
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
    >>> float(lb_keogh(ts_query=ts2,
    ...                envelope_candidate=(env_low, env_up)))  # doctest: +ELLIPSIS
    2.8284...
    >>> float(lb_keogh(ts_query=ts2,
    ...                ts_candidate=ts1,
    ...                radius=1))  # doctest: +ELLIPSIS
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
        assert (
            ts_candidate.shape[1] == 1
        ), "LB_Keogh is available only for univariate time series"
        envelope_down, envelope_up = lb_envelope(ts_candidate, radius)
    ts_query = to_time_series(ts_query)
    assert (
        ts_query.shape[1] == 1
    ), "LB_Keogh is available only for univariate time series"
    indices_up = ts_query[:, 0] > envelope_up[:, 0]
    indices_down = ts_query[:, 0] < envelope_down[:, 0]
    return numpy.sqrt(
        numpy.linalg.norm(ts_query[indices_up, 0] - envelope_up[indices_up, 0]) ** 2
        + numpy.linalg.norm(ts_query[indices_down, 0] - envelope_down[indices_down, 0])
        ** 2
    )


@njit()
def _njit_lb_envelope(time_series, radius):
    """Compute time series envelope as required by LB_Keogh.

    Parameters
    ----------
    time_series : array-like, shape=(sz, d)
        Time series for which the envelope should be computed.
    radius : int
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on all observations from the time series
        at indices comprised between i-radius and i+radius).

    Returns
    -------
    envelope_down : array-like, shape=(sz, d)
        Lower-side of the envelope.
    envelope_up : array-like, shape=(sz, d)
        Upper-side of the envelope.
    """
    sz, d = time_series.shape
    envelope_up = numpy.empty((sz, d))
    envelope_down = numpy.empty((sz, d))

    for i in prange(sz):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        for di in prange(d):
            envelope_down[i, di] = numpy.min(time_series[min_idx:max_idx, di])
            envelope_up[i, di] = numpy.max(time_series[min_idx:max_idx, di])

    return envelope_down, envelope_up


def _lb_envelope(time_series, radius, be=None):
    """Compute time series envelope as required by LB_Keogh.

    Parameters
    ----------
    time_series : array-like, shape=(sz, d)
        Time series for which the envelope should be computed.
    radius : int
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on all observations from the time series
        at indices comprised between i-radius and i+radius).
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    envelope_down : array-like, shape=(sz, d)
        Lower-side of the envelope.
    envelope_up : array-like, shape=(sz, d)
        Upper-side of the envelope.
    """
    be = instantiate_backend(be, time_series)
    time_series = be.array(time_series)
    sz, d = be.shape(time_series)
    envelope_up = be.empty((sz, d))
    envelope_down = be.empty((sz, d))

    for i in range(sz):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        for di in range(d):
            envelope_down[i, di] = be.min(time_series[min_idx:max_idx, di])
            envelope_up[i, di] = be.max(time_series[min_idx:max_idx, di])

    return envelope_down, envelope_up


def lb_envelope(ts, radius=1, be=None):
    r"""Compute time series envelope as required by LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts : array-like, shape=(sz, d) or (sz,)
        Time series for which the envelope should be computed.
        If shape is (sz,), the time series is assumed to be univariate.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on all observations from the time series
        at indices comprised between i-radius and i+radius).
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    envelope_down : array-like, shape=(sz, d)
        Lower-side of the envelope.
    envelope_up : array-like, shape=(sz, d)
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
    be = instantiate_backend(be, ts)
    ts = to_time_series(ts, be=be)
    if be.is_numpy:
        return _njit_lb_envelope(ts, radius=radius)
    return _lb_envelope(ts, radius=radius, be=be)


@njit(nogil=True)
def njit_lcss_accumulated_matrix(s1, s2, eps, mask):
    """Compute the longest common subsequence similarity score between
    two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        First time series.
    s2 : array-like, shape=(sz2, d)
        Second time series.
    eps : float
        Matching threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
        
    Returns
    -------
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    acc_cost_mat = numpy.full((l1 + 1, l2 + 1), 0)

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if numpy.isfinite(mask[i - 1, j - 1]):
                if numpy.sqrt(_njit_local_squared_dist(s1[i - 1], s2[j - 1])) <= eps:
                    acc_cost_mat[i][j] = 1 + acc_cost_mat[i - 1][j - 1]
                else:
                    acc_cost_mat[i][j] = max(
                        acc_cost_mat[i][j - 1], acc_cost_mat[i - 1][j]
                    )

    return acc_cost_mat


def lcss_accumulated_matrix(s1, s2, eps, mask, be=None):
    """Compute the longest common subsequence similarity score between
    two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        First time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Second time series. If shape is (sz2,), the time series is assumed to be univariate.
    eps : float
        Matching threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)
    l1 = be.shape(s1)[0]
    l2 = be.shape(s2)[0]
    acc_cost_mat = be.full((l1 + 1, l2 + 1), 0)

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if mask[i - 1, j - 1]:
                if be.is_numpy:
                    squared_dist = _njit_local_squared_dist(s1[i - 1], s2[j - 1])
                else:
                    squared_dist = _local_squared_dist(s1[i - 1], s2[j - 1], be=be)
                if be.sqrt(squared_dist) <= eps:
                    acc_cost_mat[i][j] = 1 + acc_cost_mat[i - 1][j - 1]
                else:
                    acc_cost_mat[i][j] = max(
                        acc_cost_mat[i][j - 1], acc_cost_mat[i - 1][j]
                    )

    return acc_cost_mat


@njit(nogil=True)
def _njit_lcss(s1, s2, eps, mask):
    """Compute the longest common subsequence score between two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        First time series.
    s2 : array-like, shape=(sz2, d)
        Second time series.
    eps : float (default: 1.)
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.

    Returns
    -------
    lcss_score : float
        Longest Common Subsquence score between both time series.
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    acc_cost_mat = njit_lcss_accumulated_matrix(s1, s2, eps, mask)

    return float(acc_cost_mat[-1][-1]) / min([l1, l2])


def _lcss(s1, s2, eps, mask, be=None):
    """Compute the longest common subsequence score between two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        First time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Second time series. If shape is (sz2,), the time series is assumed to be univariate.
    eps : float (default: 1.)
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    lcss_score : float
        Longest Common Subsquence score between both time series.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = be.array(s1)
    s2 = be.array(s2)
    l1 = be.shape(s1)[0]
    l2 = be.shape(s2)[0]
    acc_cost_mat = lcss_accumulated_matrix(s1, s2, eps, mask, be=be)
    return float(acc_cost_mat[-1][-1]) / min([l1, l2])


def lcss(
    s1,
    s2,
    eps=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute the Longest Common Subsequence (LCSS) similarity measure
    between (possibly multidimensional) time series and return the
    similarity.

    LCSS is computed by matching indexes that are met up until the eps
    threshold, so it leaves some points unmatched and focuses on the
    similar parts of two sequences. The matching can occur even if the
    time indexes are different. One can set additional constraints to the
    set of acceptable paths: the Sakoe-Chiba band which is parametrized by a
    radius or the Itakura parallelogram which is parametrized by a maximum slope.
    Both these constraints consists in forcing paths to lie close
    to the diagonal. To retrieve a meaningful similarity value from the
    length of the longest common subsequence, the percentage of that value
    regarding the length of the shortest time series is returned.

    According to this definition, the values returned by LCSS range from
    0 to 1, the highest value taken when two time series fully match,
    and vice-versa. It is not required that both time series share the
    same size, but they must be the same dimension. LCSS was originally
    presented in [1]_ and is discussed in more details in our
    :ref:`dedicated user-guide page <lcss>`.

    Notes
    -----
    Contrary to Dynamic Time Warping and variants, an LCSS path does not need to be contiguous.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    eps : float (default: 1.)
        Maximum matching distance threshold.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for LCSS.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    float
        Similarity score

    Examples
    --------
    >>> lcss([1, 2, 3], [1., 2., 2., 3.])
    1.0
    >>> lcss([1, 2, 3], [1., 2., 2., 4., 7.])
    1.0
    >>> lcss([1, 2, 3], [1., 2., 2., 2., 3.], eps=0)
    1.0
    >>> lcss([1, 2, 3], [-2., 5., 7.], eps=3)
    0.6666666666666666

    See Also
    --------
    lcss_path: Get both the matching path and the similarity score for LCSS

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
            Similar Multidimensional Trajectories", In Proceedings of the
            18th International Conference on Data Engineering (ICDE '02).
            IEEE Computer Society, USA, 673.

    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    mask = compute_mask_(
        s1,
        s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        be=be,
    )
    if be.is_numpy:
        return _njit_lcss(s1, s2, eps, mask)
    return _lcss(s1, s2, eps, mask, be=be)


@njit()
def _njit_return_lcss_path(s1, s2, eps, mask, acc_cost_mat, sz1, sz2):
    """Return the Longest Common Subsequence (LCSS) path.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        A time series.
    s2 : array-like, shape=(sz2, d)
        Another time series.
    eps : float
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    sz1 : int
        Length of the first time series.
    sz2 : int
        Length of the second time series.

    Returns
    -------
    path : list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.
    """
    i, j = (sz1, sz2)
    path = []

    while i > 0 and j > 0:
        if numpy.isfinite(mask[i - 1, j - 1]):
            if numpy.sqrt(_njit_local_squared_dist(s1[i - 1], s2[j - 1])) <= eps:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
            elif acc_cost_mat[i - 1][j] > acc_cost_mat[i][j - 1]:
                i = i - 1
            else:
                j = j - 1
    return path[::-1]


def _return_lcss_path(s1, s2, eps, mask, acc_cost_mat, sz1, sz2, be=None):
    """Return the Longest Common Subsequence (LCSS) path.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        A time series.
    s2 : array-like, shape=(sz2, d)
        Another time series.
    eps : float
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    sz1 : int
        Length of the first time series.
    sz2 : int
        Length of the second time series.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    path : list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.
    """
    be = instantiate_backend(be, s1, s2, acc_cost_mat)
    s1 = be.array(s1)
    s2 = be.array(s2)
    acc_cost_mat = be.array(acc_cost_mat)
    i, j = (sz1, sz2)
    path = []

    while i > 0 and j > 0:
        if mask[i - 1, j - 1]:
            if be.is_numpy:
                squared_dist = _njit_local_squared_dist(s1[i - 1], s2[j - 1])
            else:
                squared_dist = _local_squared_dist(s1[i - 1], s2[j - 1])
            if be.sqrt(squared_dist) <= eps:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
            elif acc_cost_mat[i - 1][j] > acc_cost_mat[i][j - 1]:
                i = i - 1
            else:
                j = j - 1
    return path[::-1]


@njit()
def _njit_return_lcss_path_from_dist_matrix(
    dist_matrix, eps, mask, acc_cost_mat, sz1, sz2
):
    """Return the Longest Common Subsequence (LCSS) path from the distance matrix.

    Parameters
    ----------
    dist_matrix : array-like, shape=(sz1, sz2)
        Matrix of the pairwise distances between the values
        of a time series s1 and the values of a time series s2.
    eps : float
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    sz1 : int
        Length of the first time series.
    sz2 : int
        Length of the second time series.

    Returns
    -------
    path : list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to a first time series s1 and the second one
        corresponds to a second time series s2.
    """
    i, j = (sz1, sz2)
    path = []

    while i > 0 and j > 0:
        if numpy.isfinite(mask[i - 1, j - 1]):
            if dist_matrix[i - 1, j - 1] <= eps:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
            elif acc_cost_mat[i - 1][j] > acc_cost_mat[i][j - 1]:
                i = i - 1
            else:
                j = j - 1
    return path[::-1]


def _return_lcss_path_from_dist_matrix(
    dist_matrix, eps, mask, acc_cost_mat, sz1, sz2, be=None
):
    """Return the Longest Common Subsequence (LCSS) path from the distance matrix.

    Parameters
    ----------
    dist_matrix : array-like, shape=(sz1, sz2)
        Matrix of the pairwise distances between the values
        of a time series s1 and the values of a time series s2.
    eps : float
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    sz1 : int
        Length of the first time series.
    sz2 : int
        Length of the second time series.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    path : list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to a first time series s1 and the second one
        corresponds to a second time series s2.
    """
    be = instantiate_backend(be, dist_matrix, acc_cost_mat)
    dist_matrix = be.array(dist_matrix)
    acc_cost_mat = be.array(acc_cost_mat)
    i, j = (sz1, sz2)
    path = []

    while i > 0 and j > 0:
        if mask[i - 1, j - 1]:
            if dist_matrix[i - 1, j - 1] <= eps:
                path.append((i - 1, j - 1))
                i, j = (i - 1, j - 1)
            elif acc_cost_mat[i - 1][j] > acc_cost_mat[i][j - 1]:
                i = i - 1
            else:
                j = j - 1
    return path[::-1]


def lcss_path(
    s1,
    s2,
    eps=1,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute the Longest Common Subsequence (LCSS) similarity measure
    between (possibly multidimensional) time series and return both the
    path and the similarity.

    LCSS is computed by matching indexes that are met up until the eps
    threshold, so it leaves some points unmatched and focuses on the
    similar parts of two sequences. The matching can occur even if the
    time indexes are different. One can set additional constraints to
    the set of acceptable paths: the Sakoe-Chiba band which is parametrized
    by a radius or the Itakura parallelogram which is parametrized by a
    maximum slope. Both these constraints consists in forcing paths to lie
    close to the diagonal.

    To retrieve a meaningful similarity value from the length of the
    longest common subsequence, the percentage of that value regarding
    the length of the shortest time series is returned.

    According to this definition, the values returned by LCSS range from
    0 to 1, the highest value taken when two time series fully match,
    and vice-versa. It is not required that both time series share the
    same size, but they must be the same dimension. LCSS was originally
    presented in [1]_ and is discussed in more details in our
    :ref:`dedicated user-guide page <lcss>`.

    Notes
    -----
    Contrary to Dynamic Time Warping and variants, an LCSS path does not need to be contiguous.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    eps : float (default: 1.)
        Maximum matching distance threshold.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for LCSS.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2

    float
        Similarity score

    Examples
    --------
    >>> path, sim = lcss_path([1., 2., 3.], [1., 2., 2., 3.])
    >>> path
    [(0, 1), (1, 2), (2, 3)]
    >>> sim
    1.0
    >>> lcss_path([1., 2., 3.], [1., 2., 2., 4.])[1]
    1.0

    See Also
    --------
    lcss : Get only the similarity score for LCSS
    lcss_path_from_metric: Compute LCSS using a user-defined distance metric

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
            Similar Multidimensional Trajectories", In Proceedings of the
            18th International Conference on Data Engineering (ICDE '02).
            IEEE Computer Society, USA, 673.

    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    mask = compute_mask_(
        s1,
        s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius,
        itakura_max_slope,
        be=be,
    )

    l1 = s1.shape[0]
    l2 = s2.shape[0]

    if be.is_numpy:
        acc_cost_mat = njit_lcss_accumulated_matrix(s1, s2, eps, mask)
    else:
        acc_cost_mat = lcss_accumulated_matrix(s1, s2, eps, mask, be=be)
    path = _return_lcss_path(s1, s2, eps, mask, acc_cost_mat, l1, l2, be=be)

    return path, float(acc_cost_mat[-1][-1]) / min([l1, l2])


@njit()
def njit_lcss_accumulated_matrix_from_dist_matrix(dist_matrix, eps, mask):
    """Compute the accumulated cost matrix score between two time series using
    a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : array-like, shape=(sz1, sz2)
        Array containing the pairwise distances.
    eps : float (default: 1.)
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.

    Returns
    -------
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    """
    l1, l2 = dist_matrix.shape
    acc_cost_mat = numpy.full((l1 + 1, l2 + 1), 0)

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if numpy.isfinite(mask[i - 1, j - 1]):
                if dist_matrix[i - 1, j - 1] <= eps:
                    acc_cost_mat[i][j] = 1 + acc_cost_mat[i - 1][j - 1]
                else:
                    acc_cost_mat[i][j] = max(
                        acc_cost_mat[i][j - 1], acc_cost_mat[i - 1][j]
                    )

    return acc_cost_mat


def lcss_accumulated_matrix_from_dist_matrix(dist_matrix, eps, mask, be=None):
    """Compute the accumulated cost matrix score between two time series using
    a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : array-like, shape=(sz1, sz2)
        Array containing the pairwise distances.
    eps : float (default: 1.)
        Maximum matching distance threshold.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have False values.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    acc_cost_mat : array-like, shape=(sz1 + 1, sz2 + 1)
        Accumulated cost matrix.
    """
    be = instantiate_backend(be, dist_matrix)
    dist_matrix = be.array(dist_matrix)
    l1, l2 = be.shape(dist_matrix)
    acc_cost_mat = be.full((l1 + 1, l2 + 1), 0)

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            if mask[i - 1, j - 1]:
                if dist_matrix[i - 1, j - 1] <= eps:
                    acc_cost_mat[i][j] = 1 + acc_cost_mat[i - 1][j - 1]
                else:
                    acc_cost_mat[i][j] = max(
                        acc_cost_mat[i][j - 1], acc_cost_mat[i - 1][j]
                    )

    return acc_cost_mat


def lcss_path_from_metric(
    s1,
    s2=None,
    eps=1,
    metric="euclidean",
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
    **kwds
):
    r"""Compute the Longest Common Subsequence (LCSS) similarity measure between
    (possibly multidimensional) time series using a distance metric defined by
    the user and return both the path and the similarity.

    Having the length of the longest commom subsequence between two time series,
    the similarity is computed as the percentage of that value regarding the
    length of the shortest time series.

    It is not required that both time series share the same size, but they must
    be the same dimension. LCSS was originally presented in [1]_.

    Valid values for metric are the same as for scikit-learn
    `pairwise_distances`_ function i.e. a string (e.g. "euclidean",
    "sqeuclidean", "hamming") or a function that is used to compute the
    pairwise distances. See `scikit`_ and `scipy`_ documentations for more
    information about the available metrics.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,) if metric!="precomputed", (sz1, sz2) otherwise
        A time series or an array of pairwise distances between samples.
        If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,), optional (default: None)
        A second time series, only allowed if metric != "precomputed".
        If shape is (sz2,), the time series is assumed to be univariate.
    eps : float (default: 1.)
        Maximum matching distance threshold.
    metric : string or callable (default: "euclidean")
        Function used to compute the pairwise distances between each points of
        `s1` and `s2`.
        If metric is "precomputed", `s1` is assumed to be a distance matrix.
        If metric is an other string, it must be one of the options compatible
        with sklearn.metrics.pairwise_distances.
        Alternatively, if metric is a callable function, it is called on pairs
        of rows of `s1` and `s2`. The callable should take two 1 dimensional
        arrays as input and return a value indicating the distance between
        them.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for LCSS.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.
    **kwds
        Additional arguments to pass to sklearn pairwise_distances to compute
        the pairwise distances.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.
    float
        Similarity score.

    Examples
    --------
    Lets create 2 numpy arrays to wrap:

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> s1, s2 = rng.rand(5, 2), rng.rand(6, 2)

    The wrapping can be done by passing a string indicating the metric to pass
    to scikit-learn pairwise_distances:

    >>> lcss_path_from_metric(s1, s2,
    ...                      metric="sqeuclidean")  # doctest: +ELLIPSIS
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.0)

    Or by defining a custom distance function:

    >>> sqeuclidean = lambda x, y: np.sum((x-y)**2)
    >>> lcss_path_from_metric(s1, s2, metric=sqeuclidean)  # doctest: +ELLIPSIS
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.0)

    Or by using a precomputed distance matrix as input:

    >>> from sklearn.metrics.pairwise import pairwise_distances
    >>> dist_matrix = pairwise_distances(s1, s2, metric="sqeuclidean")
    >>> lcss_path_from_metric(dist_matrix,
    ...                      metric="precomputed")  # doctest: +ELLIPSIS
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.0)

    Notes
    -----
    By using a squared euclidean distance metric as shown above, the output
    path and similarity is the same as the one obtained by using lcss_path
    (which uses the euclidean distance) simply because with the sum of squared
    distances the matching threshold is still not reached.
    Also, contrary to Dynamic Time Warping and variants, an LCSS path does not need to be contiguous.

    See Also
    --------
    lcss: Get only the similarity score for LCSS
    lcss_path : Get both the matching path and the similarity score for LCSS

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
            Similar Multidimensional Trajectories", In Proceedings of the
            18th International Conference on Data Engineering (ICDE '02).
            IEEE Computer Society, USA, 673.

    .. _pairwise_distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    .. _scikit: https://scikit-learn.org/stable/modules/metrics.html

    .. _scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    """  # noqa: E501
    be = instantiate_backend(be, s1, s2)
    if metric == "precomputed":  # Pairwise distance given as input
        s1 = be.array(s1)
        sz1, sz2 = be.shape(s1)
        mask = compute_mask_(
            sz1,
            sz2,
            GLOBAL_CONSTRAINT_CODE[global_constraint],
            sakoe_chiba_radius,
            itakura_max_slope,
            be=be,
        )
        dist_mat = s1
    else:
        s1 = to_time_series(s1, remove_nans=True, be=be)
        s2 = to_time_series(s2, remove_nans=True, be=be)
        sz1 = be.shape(s1)[0]
        sz2 = be.shape(s2)[0]
        mask = compute_mask_(
            s1,
            s2,
            GLOBAL_CONSTRAINT_CODE[global_constraint],
            sakoe_chiba_radius,
            itakura_max_slope,
            be=be,
        )
        dist_mat = be.array(be.pairwise_distances(s1, s2, metric=metric, **kwds))

    if be.is_numpy:
        acc_cost_mat = njit_lcss_accumulated_matrix_from_dist_matrix(
            dist_mat, eps, mask
        )
        path = _njit_return_lcss_path_from_dist_matrix(
            dist_mat, eps, mask, acc_cost_mat, sz1, sz2
        )
    else:
        acc_cost_mat = lcss_accumulated_matrix_from_dist_matrix(
            dist_mat, eps, mask, be=be
        )
        path = _return_lcss_path_from_dist_matrix(
            dist_mat, eps, mask, acc_cost_mat, sz1, sz2, be=be
        )

    return path, float(acc_cost_mat[-1][-1]) / min([sz1, sz2])
