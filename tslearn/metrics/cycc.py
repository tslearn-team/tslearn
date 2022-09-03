STUFF_cycc = "cycc"

import numpy as np
from numba import jit, njit, objmode, prange, float64, boolean

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


"""njit --> Ok
"""


# @njit(parallel=True)
@njit(float64[:](float64[:, :], float64[:, :], float64, float64), parallel=True, fastmath=True)
def normalized_cc(s1, s2, norm1=-1.0, norm2=-1.0):
    """Normalize cc.

    Parameters
    ----------
    s1 : array-like, shape=[sz, d]
    s2 : array-like, shape=[sz, d]
    norm1 : float, default=-1.0
    norm2 : float, default=-1.0

    Returns
    -------
    norm_cc : array-like, shape=[sz]
    """
    # assert s1.dtype == np.float64 and s2.dtype == np.float64
    assert s1.shape[1] == s2.shape[1]
    sz = s1.shape[0]
    # Compute fft size based on tip from https://stackoverflow.com/questions/14267555/
    # fft_sz = 1 << (2 * sz - 1).bit_length()
    n_bits = 1 + int(np.log2(2 * sz - 1))
    fft_sz = 2 ** n_bits

    if norm1 < 0.0:
        norm1 = np.linalg.norm(s1)
    if norm2 < 0.0:
        norm2 = np.linalg.norm(s2)

    denom = norm1 * norm2
    if denom < 1e-9:  # To avoid NaNs
        denom = np.inf

    with objmode(cc='float64[:, :]'):
        cc = np.real(
            np.fft.ifft(
                np.fft.fft(s1, fft_sz, axis=0) * np.conj(np.fft.fft(s2, fft_sz, axis=0)),
                axis=0,
            )
        )
    cc = np.vstack((cc[-(sz - 1) :], cc[:sz]))
    norm_cc = np.real(cc).sum(axis=-1) / denom
    return norm_cc


"""njit --> Ok
tslearn/clustering/kshape.py:148:        return 1. - cdist_normalized_cc(X, self.cluster_centers_,
"""


# @njit(parallel=True)
@njit(float64[:, :](float64[:, :, :], float64[:, :, :], float64[:], float64[:], boolean), parallel=True, fastmath=True)
def cdist_normalized_cc(dataset1, dataset2, norms1, norms2, self_similarity):
    """Compute the distance matrix between two time series dataset.

    Parameters
    ----------
    dataset1 : array-like, shape=[n_ts1, sz, d]
    dataset2 : array-like, shape=[n_ts2, sz, d]
    norms1 : array-like, shape=[n_ts1]
    norms2 : array-like, shape=[n_ts2]
    self_similarity : bool

    Returns
    -------
    dists : array-like, shape=[n_ts1, n_ts2]
    """
    n_ts1, sz, d = dataset1.shape
    n_ts2 = dataset2.shape[0]
    # assert dataset1.dtype == np.float64 and dataset2.dtype == np.float64
    assert d == dataset2.shape[2]
    dists = np.empty((n_ts1, n_ts2))

    if (norms1 < 0.0).any():
        # norms1 = np.linalg.norm(dataset1, axis=(1, 2))
        for i_ts1 in prange(n_ts1):
            norms1[i_ts1] = np.linalg.norm(dataset1[i_ts1, ...])
    if (norms2 < 0.0).any():
        # norms2 = np.linalg.norm(dataset2, axis=(1, 2))
        for i_ts2 in prange(n_ts2):
            norms2[i_ts2] = np.linalg.norm(dataset2[i_ts2, ...])
    for i in prange(n_ts1):
        for j in range(n_ts2):
            if self_similarity and j < i:
                dists[i, j] = dists[j, i]
            elif self_similarity and i == j:
                dists[i, j] = 0.0
            else:
                dists[i, j] = normalized_cc(
                    dataset1[i], dataset2[j], norm1=norms1[i], norm2=norms2[j]
                ).max()
    return dists


"""njit --> Ok
tslearn/clustering/kshape.py:120:        Xp = y_shifted_sbd_vec(self.cluster_centers_[k], X[self.labels_ == k],
"""


# @njit(parallel=True)
@njit(float64[:, :, :](float64[:, :], float64[:, :, :], float64, float64[:]), parallel=True, fastmath=True)
def y_shifted_sbd_vec(ref_ts, dataset, norm_ref, norms_dataset):
    """Shift a time series dataset w.r.t. a time series of reference.

    Parameters
    ----------
    ref_ts : array-like, shape=[sz, d]
        Time series of reference.
    dataset : array-like, shape=[n_ts, sz, d]
        Time series dataset.
    norm_ref : float
    norms_dataset : array-like, shape=[n_ts]
        Norms of the time series dataset.

    Returns
    -------
    dataset_shifted : array-like, shape=[n_ts, sz, d]
    """
    # assert dataset.dtype == np.float64 and ref_ts.dtype == np.float64
    n_ts = dataset.shape[0]
    sz = dataset.shape[1]
    d = dataset.shape[2]
    assert sz == ref_ts.shape[0] and d == ref_ts.shape[1]
    dataset_shifted = np.zeros((n_ts, sz, d))

    if norm_ref < 0:
        norm_ref = np.linalg.norm(ref_ts)
    if (norms_dataset < 0.0).any():
        # norms_dataset = np.linalg.norm(dataset, axis=(1, 2))
        for i_ts in prange(n_ts):
            norms_dataset[i_ts] = np.linalg.norm(dataset[i_ts, ...])

    for i in prange(n_ts):
        cc = normalized_cc(ref_ts, dataset[i], norm1=norm_ref, norm2=norms_dataset[i])
        idx = np.argmax(cc)
        shift = idx - sz
        if shift > 0:
            dataset_shifted[i, shift:] = dataset[i, :-shift, :]
        elif shift < 0:
            dataset_shifted[i, :shift] = dataset[i, -shift:, :]
        else:
            dataset_shifted[i] = dataset[i]

    return dataset_shifted
