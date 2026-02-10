from joblib import Parallel, delayed

from numba import njit, prange

import numpy

from tslearn.backend import instantiate_backend
from tslearn.backend.pytorch_backend import HAS_TORCH
from tslearn.utils.utils import _to_time_series

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


def __make_compute_path(backend):

    def _compute_path_generic(acc_cost_mat):
        sz1, sz2 = acc_cost_mat.shape
        path = [(sz1 - 1, sz2 - 1)]
        while path[-1] != (0, 0):
            i, j = path[-1]
            if i == 0:
                path.append((0, j - 1))
            elif j == 0:
                path.append((i - 1, 0))
            else:
                arr = backend.array(
                    [
                        acc_cost_mat[i - 1][j - 1],
                        acc_cost_mat[i - 1][j],
                        acc_cost_mat[i][j - 1],
                    ]
                )
                argmin = backend.argmin(arr)
                if argmin == 0:
                    path.append((i - 1, j - 1))
                elif argmin == 1:
                    path.append((i - 1, j))
                else:
                    path.append((i, j - 1))
        return path[::-1]
    if backend is numpy:
        return njit(nogil=True)(_compute_path_generic)
    else:
        return _compute_path_generic

_njit_compute_path = __make_compute_path(numpy)
if HAS_TORCH:
    _compute_path = __make_compute_path(instantiate_backend("torch"))
else:
    _compute_path = _njit_compute_path


def accumulated_matrix_from_dist_matrix(dist_matrix, mask, be=None):
    """Compute the accumulated cost matrix score between two time series using
    a precomputed distance matrix.

    Parameters
    ----------
    dist_matrix : array-like, shape=(sz1, sz2)
        Array containing the pairwise distances.
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
    mat : array-like, shape=(sz1, sz2)
        Accumulated cost matrix.
    """
    be = instantiate_backend(be, dist_matrix)
    dist_matrix = be.array(dist_matrix)
    if be.isnumpy:
        acc_matrix_from_dist_matrix_ = _njit_acc_matrix_from_dist_matrix
    else:
        acc_matrix_from_dist_matrix_ = _acc_matrix_from_dist_matrix

    return acc_matrix_from_dist_matrix_(dist_matrix, mask)


def __make_acc_matrix_from_dist_matrix(backend):
    if backend is numpy:
        range_ = prange
    else:
        range_ = range

    def _acc_matrix_from_dist_matrix_generic(dist_matrix, mask):
        l1, l2 = dist_matrix.shape
        cum_sum = backend.full((l1 + 1, l2 + 1), backend.inf)
        cum_sum[0, 0] = 0.0

        for i in range_(l1):
            for j in range_(l2):
                if mask[i, j]:
                    cum_sum[i + 1, j + 1] = max(
                        dist_matrix[i, j],
                        min(cum_sum[i, j + 1],
                            cum_sum[i + 1, j],
                            cum_sum[i, j])
                    )
        return cum_sum[1:, 1:]
    if backend is numpy:
        return njit(nogil=True)(_acc_matrix_from_dist_matrix_generic)
    else:
        return _acc_matrix_from_dist_matrix_generic

_njit_acc_matrix_from_dist_matrix = __make_acc_matrix_from_dist_matrix(numpy)
if HAS_TORCH:
    _acc_matrix_from_dist_matrix = __make_acc_matrix_from_dist_matrix(instantiate_backend("torch"))
else:
    _acc_matrix_from_dist_matrix = _njit_acc_matrix_from_dist_matrix



def _cdist_generic(
    dist_fun,
    dataset1,
    dataset2,
    n_jobs,
    verbose,
    be,
    compute_diagonal=True,
    dtype=float,
    *args,
    **kwargs
):
    """Compute cross-similarity matrix with joblib parallelization for a given
    similarity function.

    Parameters
    ----------
    dist_fun : function
        Similarity function to be used.

    dataset1 : array-like, shape=(n_ts1, sz1, d) or (n_ts1, sz1) or (sz1,)
        A dataset of time series.
        If shape is (n_ts1, sz1), the dataset is composed of univariate time series.
        If shape is (sz1,), the dataset is composed of a unique univariate time series.

    dataset2 : None or array-like, shape=(n_ts2, sz2, d) or (n_ts2, sz2) or (sz2,) (default: None)
        Another dataset of time series. 
        If `None`, self-similarity of `dataset1` is returned.
        If shape is (n_ts2, sz2), the dataset is composed of univariate time series.
        If shape is (sz2,), the dataset is composed of a unique univariate time series.

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

    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    compute_diagonal : bool (default: True)
        Whether diagonal terms should be computed or assumed to be 0 in the
        self-similarity case. Used only if `dataset2` is `None`.

    *args and **kwargs :
        Optional additional parameters to be passed to the similarity function.


    Returns
    -------
    cdist : array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.
    """  # noqa: E501
    n_ts_1 = len(dataset1)
    use_parallel = n_jobs not in [None, 1]

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = be.zeros((n_ts_1, n_ts_1), dtype=dtype)
        indices = be.triu_indices(
            n_ts_1, k=0 if compute_diagonal else 1, m=n_ts_1
        )

        if use_parallel:
            cdists = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
                delayed(dist_fun)(
                    _to_time_series(dataset1[i], True, be),
                    _to_time_series(dataset1[j], True, be),
                    *args,
                    **kwargs
                )
                for i in range(n_ts_1)
                for j in range(i if compute_diagonal else i + 1, n_ts_1)
            )
        else:
             cdists = [
                dist_fun(
                    _to_time_series(dataset1[i], True, be),
                    _to_time_series(dataset1[j], True, be),
                    *args,
                    **kwargs
                )
                for i in range(n_ts_1)
                for j in range(i if compute_diagonal else i + 1, n_ts_1)
            ]

        matrix[indices] = be.array(cdists, dtype=dtype)
        indices = be.tril_indices(n_ts_1, k=-1, m=n_ts_1)
        matrix[indices] = matrix.T[indices]

        return matrix
    else:
        n_ts_2 = len(dataset2)

        if use_parallel:
            cdists = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
                delayed(dist_fun)(
                    _to_time_series(dataset1[i], True, be),
                    _to_time_series(dataset2[j], True, be),
                    *args,
                    **kwargs
                )
                for i in range(n_ts_1)
                for j in range(n_ts_2)
            )
        else:
            cdists = [
                dist_fun(
                    _to_time_series(dataset1[i], True, be),
                    _to_time_series(dataset2[j], True, be),
                    *args,
                    **kwargs
                )
                for i in range(n_ts_1)
                for j in range(n_ts_2)
            ]
        matrix = be.array(cdists, dtype=dtype).reshape(n_ts_1, n_ts_2)
        return matrix
