"""Soft-DTW metric toolbox."""
import functools
import math

from joblib import Parallel, delayed

from numba import njit

import numpy

try:
    import torch
except ImportError:
    torch = None

from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.utils.utils import _to_time_series

from ._dtw import (
    _njit_dtw,
    _dtw,
    _cdist_dtw,
    _dtw_path,
    _njit_dtw_path
)
from ._masks import (
    GLOBAL_CONSTRAINT_CODE,
    _compute_mask,
    _njit_compute_mask
)
from .soft_dtw_fast import _njit_softmin3
from .utils import (
    _cdist_generic,
    _torch_accumulated_matrix,
    _torch_acc_matrix_from_dist_matrix
)


def soft_dtw(
        s1,
        s2,
        gamma=1.0,
        global_constraint=None,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
        be=None
):
    r"""Compute Soft-DTW metric between two time series.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw-softdtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series.
        If shape is (sz1,), the time series is assumed to be univariate.

    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series.
        If shape is (sz2,), the time series is assumed to be univariate.

    gamma : float (default 1.)
        Gamma parameter for Soft-DTW.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for soft-DTW.

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

    be : Backend object or string or None (default: None)
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    float
        Similarity

    Examples
    --------
    >>> soft_dtw([1, 2, 2, 3],
    ...          [1., 2., 3., 4.],
    ...          gamma=1.) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    -0.89...
    >>> soft_dtw([1, 2, 3, 3],
    ...          [1., 2., 2.1, 3.2],
    ...          gamma=0.01) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    0.09...

    The PyTorch backend can be used to compute gradients:

    >>> import torch
    >>> s1 = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    >>> s2 = torch.tensor([[3.0], [4.0], [-3.0]])
    >>> sim = soft_dtw(s1, s2, gamma=1.0)
    >>> print(sim)
    tensor(41.1876, dtype=torch.float64, grad_fn=<SelectBackward0>)
    >>> sim.backward()
    >>> print(s1.grad)
    tensor([[-4.0001],
            [-2.2852],
            [10.1643]])

    >>> s1_2d = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
    >>> s2_2d = torch.tensor([[3.0, 3.0], [4.0, 4.0], [-3.0, -3.0]])
    >>> sim = soft_dtw(s1_2d, s2_2d, gamma=1.0)
    >>> print(sim)
    tensor(83.2951, dtype=torch.float64, grad_fn=<SelectBackward0>)
    >>> sim.backward()
    >>> print(s1_2d.grad)
    tensor([[-4.0000, -4.0000],
            [-2.0261, -2.0261],
            [10.0206, 10.0206]])

    See Also
    --------
    soft_dtw_normalized: Computes the normalized similarity score for Soft-DTW
    soft_dtw_alignment: Computes both the similarity score and
        the alignment matrix and for Soft-DTW
    cdist_soft_dtw : Cross similarity matrix between time series datasets
    cdist_soft_dtw_normalized : Cross similarity matrix between time series
        datasets using a normalized version of Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
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

    if math.isclose(gamma, 0.0):
        dtw_ = _njit_dtw if be.is_numpy else _dtw
        return dtw_(s1, s2, global_constraint_, sakoe_chiba_radius, itakura_max_slope) ** 2

    soft_dtw_ = _njit_soft_dtw if be.is_numpy else _soft_dtw
    return soft_dtw_(
        s1,
        s2,
        gamma,
        global_constraint_,
        sakoe_chiba_radius,
        itakura_max_slope,
    )


def soft_dtw_normalized(
        s1,
        s2,
        gamma=1.0,
        global_constraint=None,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
        be=None
):
    r"""Compute normalized Soft-DTW metric between two time series.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw-softdtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

    This normalized version is defined as:

    .. math::

        \text{norm-soft-DTW}_{\gamma}(X, Y) =
            \text{soft-DTW}_{\gamma}(X, Y) -
            \frac{1}{2} \left(\text{soft-DTW}_{\gamma}(X, X) +
                \text{soft-DTW}_{\gamma}(Y, Y)\right)

    and ensures that all returned values are positive and that
    :math:`\text{norm-soft-DTW}_{\gamma}(X, X) = 0`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series.
        If shape is (sz1,), the time series is assumed to be univariate.

    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series.
        If shape is (sz2,), the time series is assumed to be univariate.

    gamma : float (default 1.)
        Gamma parameter for Soft-DTW.

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

    be : Backend object or string or None (default: None)
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    float
        Similarity

    Examples
    --------
    >>> soft_dtw_normalized([1, 2, 2, 3],
    ...                     [1, 2, 2, 3],
    ...                     gamma=0.1)
    0.0
    >>> soft_dtw_normalized([1, 2, 2, 3],
    ...          [1., 2., 3., 4.],
    ...          gamma=1.) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    1.309...

    See Also
    --------
    soft_dtw : Computes the similarity score for Soft-DTW
    soft_dtw_alignment: Computes both the similarity score and
        the alignment matrix and for Soft-DTW
    cdist_soft_dtw : Cross similarity matrix between time series datasets
    cdist_soft_dtw_normalized : Cross similarity matrix between time series
        datasets using a normalized version of Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
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

    if math.isclose(gamma, 0.0):
        dtw_ = _njit_dtw if be.is_numpy else _dtw
        return dtw_(s1, s2, global_constraint_, sakoe_chiba_radius, itakura_max_slope) ** 2

    soft_dtw_normalized_ = _njit_soft_dtw_normalized if be.is_numpy else _soft_dtw_normalized
    return soft_dtw_normalized_(
        s1,
        s2,
        gamma,
        global_constraint_,
        sakoe_chiba_radius,
        itakura_max_slope,
    )


@njit(nogil=True)
def _njit_accumulated_matrix(s1, s2, mask, gamma):
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
                cum_sum[i + 1, j + 1] += _njit_softmin3(
                    cum_sum[i, j + 1],
                    cum_sum[i + 1, j],
                    cum_sum[i, j],
                    gamma
                )
    return cum_sum[1:, 1:]


if torch is not None:

    def __acc_fun(distances, predecessors, gamma):
        return distances + -gamma * torch.logsumexp(-predecessors/gamma, dim=0)

    _accumulated_matrix = functools.partial(
        _torch_accumulated_matrix,
        acc_fun=__acc_fun
    )
else:
    _accumulated_matrix = _njit_accumulated_matrix


@njit(nogil=True)
def _njit_accumulated_matrix_from_distance_matrix(D, mask, gamma, out=None):
    l1, l2 = D.shape
    if out is not None:
        cum_sum = out
        out[...] = numpy.inf
    else:
        cum_sum = numpy.full((l1, l2), numpy.inf, dtype=D.dtype)

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                cum_sum[i, j] = D[i, j]
                if (i + j) != 0:
                    cum_sum[i, j] += _njit_softmin3(
                        cum_sum[i - 1, j] if i != 0 else numpy.inf,
                        cum_sum[i, j - 1] if j != 0 else numpy.inf,
                        cum_sum[i - 1, j - 1] if i*j != 0 else numpy.inf,
                        gamma
                    )

    return cum_sum


if torch is not None:

    def __acc_fun(distances, predecessors, gamma):
        return distances + -gamma * torch.logsumexp(-predecessors/gamma, dim=0)

    _accumulated_matrix_from_distance_matrix = functools.partial(
        _torch_acc_matrix_from_dist_matrix,
        acc_fun=__acc_fun
    )
else:
    _accumulated_matrix_from_distance_matrix = _njit_accumulated_matrix_from_distance_matrix


def __make_soft_dtw(backend):
    if backend is numpy:
        compute_mask_ = _njit_compute_mask
        accumulated_matrix_ = _njit_accumulated_matrix
    else:
        compute_mask_ = _compute_mask
        accumulated_matrix_ = _accumulated_matrix

    def _soft_dtw_generic(
        s1,
        s2,
        gamma,
        global_constraint=0,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
    ):
        mask = compute_mask_(s1.shape[0], s2.shape[0], global_constraint, sakoe_chiba_radius, itakura_max_slope)
        cum_sum = accumulated_matrix_(s1, s2, mask, gamma=gamma)
        return cum_sum[-1, -1]

    if backend is numpy:
        return njit(nogil=True)(_soft_dtw_generic)
    else:
        return _soft_dtw_generic


_njit_soft_dtw = __make_soft_dtw(numpy)
if torch is not None:
    _soft_dtw = __make_soft_dtw(instantiate_backend("torch"))
else:
    _soft_dtw = _njit_soft_dtw


def __make_soft_dtw_normalized(backend):
    if backend is numpy:
        soft_dtw_ = _njit_soft_dtw
    else:
        soft_dtw_ = _soft_dtw

    def _soft_dtw_normalized_generic(
            s1,
            s2,
            gamma,
            global_constraint=0,
            sakoe_chiba_radius=None,
            itakura_max_slope=None,
    ):
        return soft_dtw_(
            s1,
            s2,
            gamma,
            global_constraint,
            sakoe_chiba_radius,
            itakura_max_slope
        ) - 0.5 * (
                soft_dtw_(
                    s1,
                    s1,
                    gamma,
                    global_constraint,
                    sakoe_chiba_radius,
                    itakura_max_slope
                ) + soft_dtw_(
                    s2,
                    s2,
                    gamma,
                    global_constraint,
                    sakoe_chiba_radius,
                    itakura_max_slope
                )
        )

    if backend is numpy:
        return njit(nogil=True)(_soft_dtw_normalized_generic)
    else:
        return _soft_dtw_normalized_generic


_njit_soft_dtw_normalized = __make_soft_dtw_normalized(numpy)
if torch is not None:
    _soft_dtw_normalized = __make_soft_dtw_normalized(instantiate_backend("torch"))
else:
    _soft_dtw = _njit_soft_dtw_normalized


def soft_dtw_alignment(
    s1,
    s2,
    gamma=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None
):
    r"""Compute Soft-DTW metric between two time series and return both the
    similarity measure and the alignment matrix.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw-softdtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series.
        If shape is (sz1,), the time series is assumed to be univariate.

    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series.
        If shape is (sz2,), the time series is assumed to be univariate.

    gamma : float (default 1.)
        Gamma parameter for Soft-DTW.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for soft-DTW.

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

    be : Backend object or string or None (default: None)
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    array-like, shape=(sz1, sz2)
        Soft-alignment matrix
    float
        Similarity

    Examples
    --------
    >>> a, dist = soft_dtw_alignment([1, 2, 2, 3],
    ...                              [1., 2., 3., 4.],
    ...                              gamma=1.)  # doctest: +ELLIPSIS
    >>> float(dist)
    -0.89...
    >>> a  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[1.00...e+00, 1.88...e-01, 2.83...e-04, 4.19...e-11],
           [3.40...e-01, 8.17...e-01, 8.87...e-02, 3.94...e-05],
           [5.05...e-02, 7.09...e-01, 5.30...e-01, 6.98...e-03],
           [1.37...e-04, 1.31...e-01, 7.30...e-01, 1.00...e+00]])

    The PyTorch backend can be used to compute gradients:

    >>> import torch
    >>> s1 = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    >>> s2 = torch.tensor([[3.0], [4.0], [-3.0]])
    >>> path, sim = soft_dtw_alignment(s1, s2, gamma=1.0)
    >>> print(sim)
    tensor(41.1876, dtype=torch.float64, grad_fn=<SelectBackward0>)
    >>> sim.backward()
    >>> print(s1.grad)
    tensor([[-4.0001],
            [-2.2852],
            [10.1643]])

    >>> s1_2d = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
    >>> s2_2d = torch.tensor([[3.0, 3.0], [4.0, 4.0], [-3.0, -3.0]])
    >>> path, sim = soft_dtw_alignment(s1_2d, s2_2d, gamma=1.0)
    >>> print(sim)
    tensor(83.2951, dtype=torch.float64, grad_fn=<SelectBackward0>)
    >>> sim.backward()
    >>> print(s1_2d.grad)
    tensor([[-4.0000, -4.0000],
            [-2.0261, -2.0261],
            [10.0206, 10.0206]])

    See Also
    --------
    soft_dtw : Computes the similarity score for Soft-DTW
    soft_dtw_normalized: Computes the normalized similarity score for Soft-DTW
    cdist_soft_dtw : Cross similarity matrix between time series datasets
    cdist_soft_dtw_normalized : Cross similarity matrix between time series
        datasets using a normalized version of Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """  # noqa: E501

    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    m, n = s1.shape[0], s2.shape[0]

    if m == 0 or n == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length."
        )

    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")

    global_constraint_ = GLOBAL_CONSTRAINT_CODE[global_constraint]

    if math.isclose(gamma, 0.0):
        dtw_path_ = _njit_dtw_path if be.is_numpy else _dtw_path
        dist, path = dtw_path_(
            s1,
            s2,
            global_constraint_,
            sakoe_chiba_radius,
            itakura_max_slope
        )
        a = be.zeros((m, n))
        for i, j in path:
            a[i, j] = 1.0
        return a, dist**2

    compute_mask_ = _njit_compute_mask if be.is_numpy else _compute_mask
    mask = compute_mask_(m, n, global_constraint_, sakoe_chiba_radius, itakura_max_slope)

    D = be.cdist(s1, s2) ** 2
    D[~mask] = be.inf

    accumulated_matrix_from_distance_matrix_ = (_njit_accumulated_matrix_from_distance_matrix
                                                if be.is_numpy else _accumulated_matrix_from_distance_matrix)
    R = accumulated_matrix_from_distance_matrix_(D, mask, gamma=gamma)

    soft_dtw_grad_ = _njit_soft_dtw_grad if be.is_numpy else _soft_dtw_grad
    E = soft_dtw_grad_(D, R, gamma)

    return E, R[-1, -1]


def __make_soft_dtw_grad(backend):

    def _soft_dtw_grad_generic(D, R, gamma, out=None):
        m, n = D.shape
        if out is not None:
            E = out
        else:
            E = backend.empty((m, n), dtype=D.dtype)
        E[-1, -1] = 1

        for j in range(n, 0, -1):  # ranges from n to 1
            for i in range(m, 0, -1):  # ranges from m to 1
                if j == n and i == m:
                    continue
                r_i_j_minus_1 = R[i, j - 1] if i != m else -backend.inf
                r_i_minus_1_j = R[i - 1, j] if j != n else -backend.inf
                r_i_j = R[i, j] if i != m and j != n else -backend.inf
                d_i_j_minus_1 = D[i, j - 1] if i != m else 0
                d_i_minus_1_j = D[i - 1, j] if j != n else 0
                d_i_j = D[i, j] if i != m and j != n else 0
                a = backend.exp((r_i_j_minus_1 - R[i - 1, j - 1] - d_i_j_minus_1) / gamma)
                b = backend.exp((r_i_minus_1_j - R[i - 1, j - 1] - d_i_minus_1_j) / gamma)
                c = backend.exp((r_i_j - R[i - 1, j - 1] - d_i_j) / gamma)
                if backend.isnan(a):
                    a = 0
                if backend.isnan(b):
                    b = 0
                if backend.isnan(c):
                    c = 0
                e_i_j_minus_1 = E[i, j - 1] if i != m else 0
                e_i_minus_1_j = E[i - 1, j] if j != n else 0
                e_i_j = E[i, j] if i != m and j != n else 0
                E[i - 1, j - 1] = e_i_j_minus_1 * a + e_i_minus_1_j * b + e_i_j * c

        return E

    if backend is numpy:
        return njit(nogil=True)(_soft_dtw_grad_generic)
    else:
        return _soft_dtw_grad_generic

_njit_soft_dtw_grad = __make_soft_dtw_grad(numpy)
if torch is not None:
    _soft_dtw_grad = __make_soft_dtw_grad(torch)
else:
    _soft_dtw_grad = _njit_soft_dtw_grad


def cdist_soft_dtw(
    dataset1,
    dataset2=None,
    gamma=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None
):
    r"""Compute cross-similarity matrix using Soft-DTW metric.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw-softdtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

    Parameters
    ----------
    dataset1 : array-like, shape=(n_ts1, sz1, d) or (n_ts1, sz1) or (sz1,)
        A dataset of time series.
        If shape is (n_ts1, sz1), the dataset is composed of univariate time series.
        If shape is (sz1,), the dataset is composed of a unique univariate time series.

    dataset2 : None or array-like, shape=(n_ts2, sz2, d) or (n_ts2, sz2) or (sz2,) (default: None)
        Another dataset of time series. If `None`, self-similarity of `dataset1` is returned.
        If shape is (n_ts2, sz2), the dataset is composed of univariate time series.
        If shape is (sz2,), the dataset is composed of a unique univariate time series.

    gamma : float (default 1.)
        Gamma parameter for Soft-DTW.

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

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    be : Backend object or string or None (default: None)
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.

    Examples
    --------
    >>> cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], gamma=.01)
    array([[-0.01098612,  1.        ],
           [ 1.        ,  0.        ]])
    >>> cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]],
    ...                [[1, 2, 2, 3], [1., 2., 3., 4.]], gamma=.01)
    array([[-0.01098612,  1.        ],
           [ 1.        ,  0.        ]])

    The PyTorch backend can be used to compute gradients:

    >>> import torch
    >>> dataset1 = torch.tensor([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]], requires_grad=True)
    >>> dataset2 = torch.tensor([[[3.0], [4.0], [-3.0]], [[3.0], [4.0], [-3.0]]])
    >>> sim_mat = cdist_soft_dtw(dataset1, dataset2, gamma=1.0)
    >>> print(sim_mat)
    tensor([[41.1876, 41.1876],
            [41.1876, 41.1876]], dtype=torch.float64, grad_fn=<ViewBackward0>)
    >>> sim = sim_mat[0, 0]
    >>> sim.backward()
    >>> print(dataset1.grad)
    tensor([[[-4.0001],
             [-2.2852],
             [10.1643]],
    <BLANKLINE>
            [[ 0.0000],
             [ 0.0000],
             [ 0.0000]]])

    See Also
    --------
    soft_dtw : Computes the similarity score for Soft-DTW
    soft_dtw_normalized: Computes the normalized similarity score for Soft-DTW
    soft_dtw_alignment: Computes both the similarity score and
        the alignment matrix and for Soft-DTW
    cdist_soft_dtw_normalized : Cross similarity matrix between time series
        datasets using a normalized version of Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """  # noqa: E501

    be = instantiate_backend(be, dataset1, dataset2)
    dataset1 = to_time_series_dataset(dataset1, be=be)
    if dataset2 is not None:
        dataset2 = to_time_series_dataset(dataset2, be=be)

    return _cdist_soft_dtw(
        dataset1=dataset1,
        dataset2=dataset2,
        gamma=gamma,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
    )


def _cdist_soft_dtw(
    dataset1,
    dataset2=None,
    gamma=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    if be is None:
        be = instantiate_backend(dataset1, dataset2)

    if math.isclose(gamma, 0.0):
        return _cdist_dtw(
            dataset1=dataset1,
            dataset2=dataset2,
            global_constraint=global_constraint,
            sakoe_chiba_radius=sakoe_chiba_radius,
            itakura_max_slope=itakura_max_slope,
            n_jobs=n_jobs,
            verbose=verbose,
            be=be,
        )

    soft_dtw_ = _njit_soft_dtw if be.is_numpy else _soft_dtw
    return _cdist_generic(
        dist_fun=soft_dtw_,
        dataset1=dataset1,
        dataset2=dataset2,
        gamma=gamma,
        compute_diagonal=True,
        global_constraint=GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
    )


def cdist_soft_dtw_normalized(
    dataset1,
    dataset2=None,
    gamma=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None
):
    r"""Compute cross-similarity matrix using a normalized version of the
    Soft-DTW metric.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw-softdtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

    This normalized version is defined as:

    .. math::

        \text{norm-soft-DTW}_{\gamma}(X, Y) =
            \text{soft-DTW}_{\gamma}(X, Y) -
            \frac{1}{2} \left(\text{soft-DTW}_{\gamma}(X, X) +
                \text{soft-DTW}_{\gamma}(Y, Y)\right)

    and ensures that all returned values are positive and that
    :math:`\text{norm-soft-DTW}_{\gamma}(X, X) = 0`.

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

    gamma : float (default 1.)
        Gamma parameter for Soft-DTW.

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

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    be : Backend object or string or None (default: None)
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.

    Examples
    --------
    >>> time_series = numpy.random.randn(10, 15, 1)
    >>> bool(numpy.all(cdist_soft_dtw_normalized(time_series) >= 0.))
    True
    >>> time_series2 = numpy.random.randn(4, 15, 1)
    >>> bool(numpy.all(cdist_soft_dtw_normalized(time_series, time_series2) >= 0.))
    True

    The PyTorch backend can be used to compute gradients:

    >>> import torch
    >>> dataset1 = torch.tensor([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]], requires_grad=True)
    >>> dataset2 = torch.tensor([[[3.0], [4.0], [-3.0]], [[3.0], [4.0], [-3.0]]])
    >>> sim_mat = cdist_soft_dtw_normalized(dataset1, dataset2, gamma=1.0)
    >>> print(sim_mat)
    tensor([[42.0586, 42.0586],
            [42.0586, 42.0586]], dtype=torch.float64, grad_fn=<AddBackward0>)
    >>> sim = sim_mat[0, 0]
    >>> sim.backward()
    >>> print(dataset1.grad)
    tensor([[[-3.5249],
             [-2.2852],
             [ 9.6891]],
    <BLANKLINE>
            [[ 0.0000],
             [ 0.0000],
             [ 0.0000]]])

    See Also
    --------
    soft_dtw : Computes the similarity score for Soft-DTW
    soft_dtw_normalized: Computes the normalized similarity score for Soft-DTW
    soft_dtw_alignment: Computes both the similarity score and
        the alignment matrix and for Soft-DTW
    cdist_soft_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """  # noqa: E501

    be = instantiate_backend(be, dataset1, dataset2)
    dataset1 = to_time_series_dataset(dataset1, be=be)
    if dataset2 is not None:
        dataset2 = to_time_series_dataset(dataset2, be=be)

    return _cdist_soft_dtw_normalized(
        dataset1=dataset1,
        dataset2=dataset2,
        gamma=gamma,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
    )


def _cdist_soft_dtw_normalized(
    dataset1,
    dataset2=None,
    gamma=1.0,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    if be is None:
        be = instantiate_backend(dataset1, dataset2)

    cdist = _cdist_soft_dtw(
        dataset1=dataset1,
        dataset2=dataset2,
        gamma=gamma,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
    )

    # DTW case, no need for normalization
    if math.isclose(gamma, 0.0):
        return cdist

    if dataset2 is None:
        d_ii = be.diag(cdist)
        normalizer = -0.5 * (be.reshape(d_ii, (-1, 1)) + be.reshape(d_ii, (1, -1)))
    else:
        use_parallel = n_jobs not in [None, 1]

        soft_dtw_ = _njit_soft_dtw if be.is_numpy else _soft_dtw
        global_constraint_ = GLOBAL_CONSTRAINT_CODE[global_constraint]

        self_dists_dataset1 = be.empty((dataset1.shape[0], 1), dtype=dataset1.dtype)
        self_dists_dataset2 = be.empty((dataset2.shape[0], 1), dtype=dataset2.dtype)

        if not use_parallel:
            for self_dist, dataset in zip((self_dists_dataset1, self_dists_dataset2), (dataset1, dataset2)):
                for i, ts in enumerate(dataset):
                    ts_ = _to_time_series(ts, True, be)
                    self_dist[i, 0] = soft_dtw_(
                        ts_,
                        ts_,
                        gamma=gamma,
                        global_constraint=global_constraint_,
                        sakoe_chiba_radius=sakoe_chiba_radius,
                        itakura_max_slope=itakura_max_slope,
                    )
        else:
            def soft_dtw_variable_lenght(ts):
               ts_ = _to_time_series(ts, True, be)
               return soft_dtw_(
                    ts_,
                    ts_,
                    gamma=gamma,
                    global_constraint=global_constraint_,
                    sakoe_chiba_radius=sakoe_chiba_radius,
                    itakura_max_slope=itakura_max_slope,
            )

            self_dists = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
                delayed(soft_dtw_variable_lenght)(ts)
                for ts in be.vstack((dataset1, dataset2))
            )
            self_dists_dataset1 = be.array(self_dists[:len(dataset1)]).reshape(-1, 1)
            self_dists_dataset2 = be.array(self_dists[len(dataset1):]).reshape(-1, 1)

        normalizer = -0.5 * (self_dists_dataset1 + self_dists_dataset2.T)

    return cdist + normalizer
