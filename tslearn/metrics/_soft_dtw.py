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

from ._cuda_metrics import _dtw_cuda, _soft_dtw_cuda
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
from .soft_dtw_fast import _njit_softmin3, _soft_dtw_grad, _njit_soft_dtw_grad
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
        if be.is_numpy:
            dtw_ = _njit_dtw
        else:
            dtw_ = _dtw_cuda if (s1.device.type==s2.device.type=="cuda") else _dtw
        return dtw_(s1, s2, global_constraint, sakoe_chiba_radius, itakura_max_slope) ** 2

    if be.is_numpy:
        soft_dtw_ = _njit_soft_dtw
    else:
        soft_dtw_ = _soft_dtw_cuda if (s1.device.type==s2.device.type=="cuda") else _soft_dtw
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
        return dtw_(s1, s2, global_constraint, sakoe_chiba_radius, itakura_max_slope) ** 2

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
def _njit_accumulated_matrix_from_distance_matrix(D, mask, gamma):
    l1, l2 = D.shape
    cum_sum = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    cum_sum[0, 0] = 0.0

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                cum_sum[i + 1, j + 1] = D[i, j]
                cum_sum[i + 1, j + 1] += _njit_softmin3(
                    cum_sum[i, j + 1],
                    cum_sum[i + 1, j],
                    cum_sum[i, j],
                    gamma
                )
    return cum_sum[1:, 1:]

if torch is not None:

    def acc_fun(distances, predecessors, gamma):
        return distances + -gamma * torch.logsumexp(predecessors * (-gamma), dim=0)

    _accumulated_matrix_from_distance_matrix = functools.partial(
        _torch_acc_matrix_from_dist_matrix,
        acc_fun=acc_fun
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
    mask = compute_mask_(m, n, global_constraint, sakoe_chiba_radius, itakura_max_slope)

    D = be.cdist(s1, s2)
    D[~mask] = be.inf

    accumulated_matrix_from_distance_matrix_ = (_njit_accumulated_matrix_from_distance_matrix
                                                if be.is_numpy else _accumulated_matrix_from_distance_matrix)
    R = accumulated_matrix_from_distance_matrix_(D, mask, gamma=gamma)

    D = be.vstack((D, be.zeros(n)))
    D = be.hstack((D, be.zeros((m + 1, 1))))
    R = be.vstack((be.zeros(n), R))
    R = be.hstack((be.zeros((m + 1, 1)), R))
    R = be.vstack((R, be.zeros(n + 1)))
    R = be.hstack((R, be.zeros((m + 2, 1))))

    soft_dtw_grad_ = (_njit_soft_dtw_grad if be.is_numpy else
                      functools.partial(_soft_dtw_grad, be=be))
    E = be.zeros((m + 2, n + 2), dtype=D.dtype)
    soft_dtw_grad_(D, R, E, gamma)

    return E[1:-1, 1:-1], R[-2, -2]


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

# Test cdist_soft_dtw_normalized with gamma = 0 is same as
# cdist_soft_dtw
# cdist_dtw


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
            def _(ts):
               ts_ =  _to_time_series(ts, True, be)
               return soft_dtw_(
                    ts_,
                    ts_,
                    gamma=gamma,
                    global_constraint=global_constraint_,
                    sakoe_chiba_radius=sakoe_chiba_radius,
                    itakura_max_slope=itakura_max_slope,
            )

            self_dists = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
                delayed(_)(
                    ts
                )
                for ts in be.hstack((dataset1, dataset2))
            )
            self_dists_dataset1 = be.array(self_dists[:len(dataset1)]).reshape(-1, 1)
            self_dists_dataset2 = be.array(self_dists[len(dataset1):]).reshape(1, -1)

        normalizer = -0.5 * (self_dists_dataset1 + self_dists_dataset2.T)

    return cdist + normalizer
