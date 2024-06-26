import warnings

import numpy
from numba import njit, prange

from tslearn.backend import instantiate_backend, cast, Backend
from tslearn.utils import to_time_series

from .utils import _cdist_generic
from tslearn.metrics.dtw_variants import _njit_local_squared_dist, _local_squared_dist, compute_mask, _return_path, _njit_return_path

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}


@njit()
def _njit_accumulated_matrix(s1, s2, mask):
    """Compute the accumulated cost matrix score between two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        First time series.
    s2 : array-like, shape=(sz2, d)
        Second time series.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.0

    for i in range(l1):
        for j in range(l2):
            if numpy.isfinite(mask[i, j]):
                cum_sum[i + 1, j + 1] = max(
                    _njit_local_squared_dist(s1[i], s2[j]),
                    min(cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j])
                )
    return numpy.sqrt(cum_sum[1:, 1:])

def _accumulated_matrix(s1, s2, mask, be=None):
    """Compute the accumulated cost matrix score between two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        First time series.
    s2 : array-like, shape=(sz2, d)
        Second time series.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have infinite values.
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
    be = instantiate_backend(be, s1, s2)
    s1 = be.array(s1)
    s2 = be.array(s2)
    l1 = be.shape(s1)[0]
    l2 = be.shape(s2)[0]
    cum_sum = be.full((l1 + 1, l2 + 1), be.inf)
    cum_sum[0, 0] = 0.0

    for i in range(l1):
        for j in range(l2):
            if be.isfinite(mask[i, j]):
                cum_sum[i + 1, j + 1] = max(
                    _local_squared_dist(s1[i], s2[j], be=be),
                    min(cum_sum[i, j + 1], cum_sum[i + 1, j], cum_sum[i, j])
                )
    return be.sqrt(cum_sum[1:, 1:])

@njit(nogil=True)
def _njit_frechet(s1, s2, mask):
    """Compute the Frechet score between two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        First time series.
    s2 : array-like, shape=(sz2, d)
        Second time series.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have infinite values.

    Returns
    -------
    frechet_score : float
        Frechet score between both time series.

    """
    cum_sum = _njit_accumulated_matrix(s1, s2, mask)
    return cum_sum[-1, -1]


def _frechet(s1, s2, mask, be=None):
    """Compute the frechet score between two time series.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d)
        First time series.
    s2 : array-like, shape=(sz2, d)
        Second time series.
    mask : array-like, shape=(sz1, sz2)
        Mask. Unconsidered cells must have infinite values.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    frechet_score : float
        Frechet score between both time series.

    """
    be = instantiate_backend(be, s1, s2)
    s1 = be.array(s1)
    s2 = be.array(s2)
    cum_sum = _accumulated_matrix(s1, s2, mask, be=be)
    return cum_sum[-1, -1]


@njit()
def _njit_return_pathpair(acc_cost_mat):
    """Return max pair from the best path from accumulated cost matrix.

    Parameters
    ----------
    acc_cost_mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.

    Returns
    -------
    max_pair : integer pair
        Pair of indices corresponding to the pair of point from which the frechet similarity comes from.
    """
    sz1, sz2 = acc_cost_mat.shape
    i, j = (sz1 - 1, sz2 - 1)
    path = [(i, j)]
    pair = (i, j)
    while (i, j) != (0, 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
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
                i, j = i - 1, j - 1
            elif argmin == 1:
                i = i - 1
            else:
                j = j - 1
        if acc_cost_mat[i, j] == acc_cost_mat[-1, -1]:
            pair = (i, j)
        path.append((i, j))
    return path[::-1], pair


def _return_pathpair(acc_cost_mat, be=None):
    """Return most separated pair from the optimal path from accumulated cost matrix.

    Parameters
    ----------
    acc_cost_mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    pair : integer pair
        Pair of indices corresponding to the pair of point from which the frechet similarity value comes from.
    """
    be = instantiate_backend(be, acc_cost_mat)
    sz1, sz2 = be.shape(acc_cost_mat)
    i, j = (sz1 - 1, sz2 - 1)
    path = [(i, j)]
    pair = (i, j)
    while (i, j) != (0, 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
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
                i, j = i - 1, j - 1
            elif argmin == 1:
                i = i - 1
            else:
                j = j - 1
        if acc_cost_mat[i, j] == acc_cost_mat[-1, -1]:
            pair = (i, j)
        path.append((i, j))
    return path[::-1], pair


def frechet(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute Frechet similarity measure between
    (possibly multidimensional) time series and return it.

    Frechet is computed as the infimum distance between optimally aligned time series,
    i.e., if :math:`\Pi` is the set of all the alignment paths (sequences of consecutive pairs of indices):

    .. math::

        \text{Frechet}(X, Y) = \min_{\pi \in \Pi} \max_{(i, j) \in \pi} \|X_{i} - Y_{j}\|

    It is not required that both time series share the same size, but they must
    be the same dimensionality to compute the norm. The paths explored are the same as in the DTW distance,
    that was originally presented in [1]_ and is discussed in more detail in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.

    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible alignement paths.

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
    >>> frechet([1, 2, 3], [1., 2., 2., 3.])
    0.0
    >>> frechet([1, 2, 3], [1., 2., 2., 3., 4.])
    1.0
    >>> frechet([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3], [1., 2., 3., 4.])
    1.0

    See Also
    --------
    frechet_path_pair : Get the optimal path, pair and the similarity score for frechet

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if be.shape(s1)[1] != be.shape(s2)[1]:
        raise ValueError("All input time series must have the same feature size.")

    mask = compute_mask(
        s1,
        s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        be=be,
    )
    if be.is_numpy:
        return _njit_frechet(s1, s2, mask=mask)
    return _frechet(s1, s2, mask=mask, be=be)


def frechet_path_pair(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute Frechet similarity measure between (possibly multidimensional) time series and return
    an optimal alignement path, the most separated pair of points in the path, and the similarity.

    Frechet is computed as the max distance between each couple of the optimally aligned time series,
    i.e., if :math:`\Pi` are all the possible alignment paths and `d` is a distance between two points:

    .. math::

        \text{Frechet}(X, Y) = \min_{\pi \in \Pi }{\max_{(i, j) \in \pi} d(X_{i}, Y_{j})}

    It is not required that both time series share the same size, but they must
    be the same dimension. The paths explored are the same as in the dtw distance,
    that was originally presented in [1]_ and is discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths.
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
    path : list of integer pair
        An optimal alignement path

    pair : integer pair
        Pair of indices corresponding to the first most separated pair of points on the returned optimal path.

    float
        Similarity score

    Examples
    --------
    >>> path, pair, dist = frechet_path_pair([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> pair
    (0, 0)
    >>> dist
    0.0
    >>> path, pair, dist = frechet_path_pair([1, 2, 3, 4], [1., 2., 5., 2., 3., 4.])
    >>> path
    [(0, 0), (1, 1), (2, 2), (2, 3), (2, 4), (3, 5)]
    >>> pair
    (2, 2)
    >>> dist
    2.0

    See Also
    --------
    frechet : Get only the similarity score for frechet

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if be.shape(s1)[1] != be.shape(s2)[1]:
        raise ValueError("All input time series must have the same feature size.")

    mask = compute_mask(
        s1,
        s2,
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius,
        itakura_max_slope,
        be=be,
    )
    if be.is_numpy:
        acc_cost_mat = _njit_accumulated_matrix(s1, s2, mask=mask)
        path, pair = _njit_return_pathpair(acc_cost_mat)
    else:
        acc_cost_mat = _accumulated_matrix(s1, s2, mask=mask, be=be)
        path, pair = _return_pathpair(acc_cost_mat, be=be)
    return path, pair, acc_cost_mat[-1, -1]
