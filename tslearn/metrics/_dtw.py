"""DTW metric toolbox."""
import functools

from numba import njit

import numpy

try:
    import torch
except ImportError:
    torch = None

from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series, to_time_series_dataset

from ._masks import (
    GLOBAL_CONSTRAINT_CODE,
    _compute_mask,
    _njit_compute_mask
)
from .utils import (
    _njit_compute_path,
    _compute_path,
    _cdist_generic,
    _torch_accumulated_matrix,
    _torch_acc_matrix_from_dist_matrix
)


def dtw(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return it.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the optimal alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.

    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

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
    >>> dtw([1, 2, 3], [1., 2., 2., 3.])
    0.0
    >>> dtw([1, 2, 3], [1., 2., 2., 3., 4.])
    1.0

    The PyTorch backend can be used to compute gradients:

    >>> import torch
    >>> s1 = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    >>> s2 = torch.tensor([[3.0], [4.0], [-3.0]])
    >>> sim = dtw(s1, s2, be="pytorch")
    >>> print(sim)
    tensor(6.4807, dtype=torch.float64, grad_fn=<SqrtBackward0>)
    >>> sim.backward()
    >>> print(s1.grad)
    tensor([[-0.3086],
            [-0.1543],
            [ 0.7715]])

    >>> s1_2d = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
    >>> s2_2d = torch.tensor([[3.0, 3.0], [4.0, 4.0], [-3.0, -3.0]])
    >>> sim = dtw(s1_2d, s2_2d, be="pytorch")
    >>> print(sim)
    tensor(9.1652, dtype=torch.float64, grad_fn=<SqrtBackward0>)
    >>> sim.backward()
    >>> print(s1_2d.grad)
    tensor([[-0.2182, -0.2182],
            [-0.1091, -0.1091],
            [ 0.5455,  0.5455]])

    See Also
    --------
    dtw_path : Get both the matching path and the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """  # noqa: E501
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if s1.shape[0] == 0 or s2.shape[0] == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    global_constraint_ = GLOBAL_CONSTRAINT_CODE[global_constraint]

    if be.is_numpy:
        dtw_ = _njit_dtw
    else:
        dtw_ = _dtw
    return dtw_(s1, s2, global_constraint_, sakoe_chiba_radius, itakura_max_slope)


def dtw_path(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series and return both the path and the
    similarity.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} (X_{i} - Y_{j})^2}

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
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
        first index corresponds to s1 and the second one corresponds to s2.

    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_path([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> float(dist)
    0.0
    >>> float(dtw_path([1, 2, 3], [1., 2., 2., 3., 4.])[1])
    1.0

    See Also
    --------
    dtw : Get only the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets
    dtw_path_from_metric : Compute a DTW using a user-defined distance metric

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    if s1.shape[0] == 0 or s2.shape[0] == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    global_constraint_ = GLOBAL_CONSTRAINT_CODE[global_constraint]

    if be.is_numpy:
        dtw_path_ = _njit_dtw_path
    else:
        dtw_path_ = _dtw_path
    dist, path = dtw_path_(
        s1,
        s2,
        global_constraint_,
        sakoe_chiba_radius,
        itakura_max_slope
    )
    return path, dist


def accumulated_matrix(s1, s2, mask, be=None):
    """Compute the DTW accumulated cost matrix score between two time series.

    It is not required that both time series share the same size, but they must
    be the same dimension.

    Parameters
    ----------
    s1 : array-like, shape=(sz1,) or (sz1, d)
        First time series.
    s2 : array-like, shape=(sz2,) or (sz2, d)
        Second time series.
    mask : array-like, shape=(sz1, sz2)
        Mask used to constrain the region of computation. Unconsidered cells must have False values.
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
        Accumulated cost matrix. Non computed cells due to masking have infinite value.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, be=be)
    s2 = to_time_series(s2, be=be)

    if be.is_numpy:
        compute_accumulated_matrix = _njit_accumulated_matrix
    else:
        compute_accumulated_matrix = _accumulated_matrix
    return compute_accumulated_matrix(s1, s2, mask)


@njit(nogil=True)
def _njit_accumulated_matrix(s1, s2, mask):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.0

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                dist = 0.0
                for di in range(s1[i].shape[0]):
                    diff = s1[i][di] - s2[j][di]
                    dist += diff * diff
                cum_sum[i + 1, j + 1] = dist
                cum_sum[i + 1, j + 1] += min(
                    cum_sum[i, j + 1],
                    cum_sum[i + 1, j],
                    cum_sum[i, j]
                )
    return cum_sum[1:, 1:]


if torch is not None:
    _accumulated_matrix = functools.partial(
        _torch_accumulated_matrix,
        acc_fun=lambda distances, predecessors: distances + torch.min(predecessors, dim=0).values
    )
else:
    _accumulated_matrix = _njit_accumulated_matrix


@njit(nogil=True)
def _njit_acc_matrix_from_dist_matrix(D, mask):
    l1, l2 = D.shape
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.0

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                cum_sum[i + 1, j + 1] = D[i, j]
                cum_sum[i + 1, j + 1] += min(
                    cum_sum[i, j + 1],
                    cum_sum[i + 1, j],
                    cum_sum[i, j]
                )
    return cum_sum[1:, 1:]


if torch is not None:
    _acc_matrix_from_dist_matrix = functools.partial(
        _torch_acc_matrix_from_dist_matrix,
        acc_fun=lambda distances, predecessors: distances + torch.min(predecessors, dim=0).values
    )
else:
    _acc_matrix_from_dist_matrix = _njit_acc_matrix_from_dist_matrix


def __make_dtw(backend):
    if backend is numpy:
        compute_mask_ = _njit_compute_mask
        accumulated_matrix_ = _njit_accumulated_matrix
    else:
        compute_mask_ = _compute_mask
        accumulated_matrix_ = _accumulated_matrix

    def _dtw_generic(
        s1,
        s2,
        global_constraint=0,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
    ):
        mask = compute_mask_(s1.shape[0], s2.shape[0], global_constraint, sakoe_chiba_radius, itakura_max_slope)
        cum_sum = accumulated_matrix_(s1, s2, mask)
        return backend.sqrt(cum_sum[-1, -1])

    if backend is numpy:
        return njit(nogil=True)(_dtw_generic)
    else:
        return _dtw_generic


_njit_dtw = __make_dtw(numpy)
if torch is not None:
    _dtw = __make_dtw(instantiate_backend("torch"))
else:
    _dtw = _njit_dtw


def __make_dtw_path(backend):
    if backend is numpy:
        compute_mask_ = _njit_compute_mask
        accumulated_matrix_ = _njit_accumulated_matrix
        compute_path_ = _njit_compute_path
    else:
        compute_mask_ = _compute_mask
        accumulated_matrix_ = _accumulated_matrix
        compute_path_ = _compute_path

    def _dtw_path_generic(
        s1,
        s2,
        global_constraint=0,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
    ):
        mask = compute_mask_(s1.shape[0], s2.shape[0], global_constraint, sakoe_chiba_radius, itakura_max_slope)
        cum_sum = accumulated_matrix_(s1, s2, mask)
        path = compute_path_(cum_sum)
        return backend.sqrt(cum_sum[-1, -1]), path

    if backend is numpy:
        return njit(nogil=True)(_dtw_path_generic)
    else:
        return _dtw_path_generic


_njit_dtw_path = __make_dtw_path(numpy)
if torch is not None:
    _dtw_path = __make_dtw_path(instantiate_backend("torch"))
else:
    _dtw_path = _njit_dtw_path


def dtw_path_from_metric(
    s1,
    s2=None,
    metric="euclidean",
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
    **kwds
):
    r"""Compute Dynamic Time Warping (DTW) similarity measure between
    (possibly multidimensional) time series using a distance metric defined by
    the user and return both the path and the similarity.

    Similarity is computed as the cumulative cost along the aligned time
    series.

    It is not required that both time series share the same size, but they must
    be the same dimension. DTW was originally presented in [1]_.

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
        Global constraint to restrict admissible paths for DTW.

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
        Similarity score (sum of metric along the wrapped time series).

    Examples
    --------
    Lets create 2 numpy arrays to wrap:

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> s1, s2 = rng.rand(5, 2), rng.rand(6, 2)

    The wrapping can be done by passing a string indicating the metric to pass
    to scikit-learn pairwise_distances:

    >>> x, y = dtw_path_from_metric(s1, s2,
    ...                             metric="sqeuclidean")  # doctest: +ELLIPSIS
    >>> x, float(y)
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Or by defining a custom distance function:

    >>> sqeuclidean = lambda x, y: np.sum((x-y)**2)
    >>> x, y = dtw_path_from_metric(s1, s2, metric=sqeuclidean)  # doctest: +ELLIPSIS
    >>> x, float(y)
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Or by using a precomputed distance matrix as input:

    >>> from sklearn.metrics.pairwise import pairwise_distances
    >>> dist_matrix = pairwise_distances(s1, s2, metric="sqeuclidean")
    >>> x, y = dtw_path_from_metric(dist_matrix,
    ...                             metric="precomputed")  # doctest: +ELLIPSIS
    >>> x, float(y)
    ([(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)], 1.117...)

    Notes
    -----
    By using a squared euclidean distance metric as shown above, the output
    path is the same as the one obtained by using dtw_path but the similarity
    score is the sum of squared distances instead of the euclidean distance.

    See Also
    --------
    dtw_path : Get both the matching path and the similarity score for DTW

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    .. _pairwise_distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    .. _scikit: https://scikit-learn.org/stable/modules/metrics.html

    .. _scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    """  # noqa: E501
    be = instantiate_backend(be, s1, s2)

    compute_mask_ = _njit_compute_mask if be.is_numpy else _compute_mask

    if metric == "precomputed":  # Pairwise distance given as input
        s1 = be.array(s1)
        sz1, sz2 = be.shape(s1)
        mask = compute_mask_(
            sz1,
            sz2,
            GLOBAL_CONSTRAINT_CODE[global_constraint],
            sakoe_chiba_radius,
            itakura_max_slope
        )
        dist_mat = s1
    else:
        s1 = to_time_series(s1, remove_nans=True, be=be)
        s2 = to_time_series(s2, remove_nans=True, be=be)
        mask = compute_mask_(
            len(s1),
            len(s2),
            GLOBAL_CONSTRAINT_CODE[global_constraint],
            sakoe_chiba_radius,
            itakura_max_slope,
        )
        dist_mat = be.pairwise_distances(s1, s2, metric=metric, **kwds)

    if be.is_numpy:
        acc_cost_mat = _njit_acc_matrix_from_dist_matrix(dist_mat, mask)
        path = _njit_compute_path(acc_cost_mat)
    else:
        dist_mat = be.array(dist_mat)
        acc_cost_mat = _acc_matrix_from_dist_matrix(dist_mat, mask)
        path = _compute_path(acc_cost_mat)
    return path, acc_cost_mat[-1, -1]


def cdist_dtw(
    dataset1,
    dataset2=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    r"""Compute cross-similarity matrix using Dynamic Time Warping (DTW)
    similarity measure.

    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`\pi` is the alignment path:

    .. math::

        DTW(X, Y) = \sqrt{\sum_{(i, j) \in \pi} \|X_{i} - Y_{j}\|^2}

    Note that this formula is still valid for the multivariate case.

    It is not required that time series share the same size, but they
    must be the same dimension.
    DTW was originally presented in [1]_ and is
    discussed in more details in our :ref:`dedicated user-guide page <dtw>`.

    Parameters
    ----------
    dataset1 : array-like, shape=(n_ts1, sz1, d) or (n_ts1, sz1) or (sz1,)
        A dataset of time series.
        If shape is (n_ts1, sz1), the dataset is composed of univariate time series.
        If shape is (sz1,), the dataset is composed of a unique univariate time series.

    dataset2 : None or array-like, shape=(n_ts2, sz2, d) or (n_ts2, sz2) or (sz2,) (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
        If shape is (n_ts2, sz2), the dataset is composed of univariate time series.
        If shape is (sz2,), the dataset is composed of a unique univariate time series.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.

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

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.

    dtype

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    cdist : array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.

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
    """  # noqa: E501
    be = instantiate_backend(be, dataset1, dataset2)
    dataset1 = to_time_series_dataset(dataset1, be=be)
    if dataset2 is not None:
        dataset2 = to_time_series_dataset(dataset2, be=be)

    return _cdist_dtw(
        dataset1=dataset1,
        dataset2=dataset2,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
    )


def _cdist_dtw(
    dataset1,
    dataset2=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    if be is None:
        be = instantiate_backend(dataset1, dataset2)
    dtw_ = _njit_dtw if be.is_numpy else _dtw
    return _cdist_generic(
        dist_fun=dtw_,
        dataset1=dataset1,
        dataset2=dataset2,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
        compute_diagonal=False,
        global_constraint=GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
    )
