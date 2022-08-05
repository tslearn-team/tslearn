STUFF_cycc = "cycc"

import numpy as np
from numba import njit, prange

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"

DTYPE = float


@njit(parallel=True)
def normalized_cc(s1, s2, norm1=-1.0, norm2=-1.0):
    assert s1.dtype == DTYPE and s2.dtype == DTYPE
    assert s1.shape[1] == s2.shape[1]
    s = 0.0
    sz = s1.shape[0]
    d = s1.shape[1]
    # Compute fft size based on tip from https://stackoverflow.com/questions/14267555/
    fft_sz = 1 << (2 * sz - 1).bit_length()
    denom = 0.0

    if norm1 < 0.0:
        norm1 = np.linalg.norm(s1)
    if norm2 < 0.0:
        norm2 = np.linalg.norm(s2)

    denom = norm1 * norm2
    if denom < 1e-9:  # To avoid NaNs
        denom = np.inf

    cc = np.real(
        np.fft.ifft(
            np.fft.fft(s1, fft_sz, axis=0) * np.conj(np.fft.fft(s2, fft_sz, axis=0)),
            axis=0,
        )
    )
    cc = np.vstack((cc[-(sz - 1) :], cc[:sz]))
    return np.real(cc).sum(axis=-1) / denom


@njit(parallel=True)
def cdist_normalized_cc(dataset1, dataset2, norms1, norms2, self_similarity):
    assert dataset1.dtype == DTYPE and dataset2.dtype == DTYPE
    assert dataset1.shape[2] == dataset2.shape[2]
    dists = np.empty((dataset1.shape[0], dataset2.shape[0]))

    if (norms1 < 0.0).any():
        norms1 = np.linalg.norm(dataset1, axis=(1, 2))
    if (norms2 < 0.0).any():
        norms2 = np.linalg.norm(dataset2, axis=(1, 2))

    for i in prange(dataset1.shape[0]):
        for j in range(dataset2.shape[0]):
            if self_similarity and j < i:
                dists[i, j] = dists[j, i]
            elif self_similarity and i == j:
                dists[i, j] = 0.0
            else:
                dists[i, j] = normalized_cc(
                    dataset1[i], dataset2[j], norm1=norms1[i], norm2=norms2[j]
                ).max()
    return dists


@njit(parallel=True)
def y_shifted_sbd_vec(ref_ts, dataset, norm_ref, norms_dataset):
    assert dataset.dtype == DTYPE and ref_ts.dtype == DTYPE
    assert dataset.shape[1] == ref_ts.shape[0] and dataset.shape[2] == ref_ts.shape[1]
    sz = dataset.shape[1]
    dataset_shifted = np.zeros((dataset.shape[0], dataset.shape[1], dataset.shape[2]))

    if norm_ref < 0:
        norm_ref = np.linalg.norm(ref_ts)
    if (norms_dataset < 0.0).any():
        norms_dataset = np.linalg.norm(dataset, axis=(1, 2))

    for i in prange(dataset.shape[0]):
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
