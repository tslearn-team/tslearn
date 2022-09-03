STUFF_cysax = "cysax"

import numpy as np
from numba import jit, njit, objmode, prange, float64, intp, int32, int64, typeof
from sklearn.linear_model import LinearRegression

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


"""njit --> Ok
tslearn/piecewise/piecewise.py:255:        return inv_transform_paa(X, original_size=self._X_fit_dims_[1])
"""


# @njit(parallel=True)
@njit(float64[:, :, :](float64[:, :, :], int32), parallel=True, fastmath=True)
def inv_transform_paa(dataset_paa, original_size):
    """Compute time series corresponding to given PAA representations.

    Parameters
    ----------
    dataset_paa : array-like, shape=[n_ts, sz, d]
        A dataset of PAA series.
    original_size : int

    Returns
    -------
    dataset_out : array-like, shape=[n_ts, original_size, d]
        A dataset of time series corresponding to the provided
        representation.
    """
    n_ts, sz, d = dataset_paa.shape
    seg_sz = original_size // sz
    dataset_out = np.zeros((n_ts, original_size, d))
    for t in prange(sz):
        t0 = t * seg_sz
        for i_ts in range(n_ts):
            dataset_out[i_ts, t0 : t0 + seg_sz, :] = dataset_paa[i_ts, t, :]
    return dataset_out


"""njit --> Ok
tslearn/piecewise/piecewise.py:451:        return cydist_sax(sax1, sax2,
"""


# @njit(parallel=True)
# @njit(float64(int64[:, :], int64[:, :], float64[:], int64))
# @njit(float64(int32[:, :], int32[:, :], float64[:], int32))
# @njit(float64(intp[:, :], intp[:, :], float64[:], intp))
# @njit(float64(typeof(1)[:, :], typeof(1)[:, :], float64[:], typeof(1)))
@njit(float64(typeof(np.array([[1], [2]])), typeof(np.array([[1], [2]])), float64[:], typeof(1)), parallel=True, fastmath=True)
def cydist_sax(sax1, sax2, breakpoints, original_size):
    """Compute distance between SAX representations as defined in [1]_.

    Parameters
    ----------
    sax1 : array-like, shape=[sz, d]
        SAX representation of a time series.
    sax2 : array-like, shape=[sz, d]
        SAX representation of another time series.
    breakpoints : array-like, ndim=1
    original_size : int

    Returns
    -------
    dist_sax : float
        SAX distance.

    References
    ----------
    .. [1] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel
       symbolic representation of time series.
       Data Mining and Knowledge Discovery, 2007. vol. 15(107)
    """
    assert sax1.shape == sax2.shape
    s = 0.0
    sz, d = sax1.shape
    for t in prange(sz):
        for di in range(d):
            if np.abs(sax1[t, di] - sax2[t, di]) > 1:
                max_symbol = max(sax1[t, di], sax2[t, di])
                min_symbol = min(sax1[t, di], sax2[t, di])
                s += (breakpoints[max_symbol - 1] - breakpoints[min_symbol]) ** 2
    dist_sax = np.sqrt(s * float(original_size) / sz)

    return dist_sax


"""njit --> Ok
tslearn/piecewise/piecewise.py:496:        X_orig = inv_transform_sax(
"""


# @njit(parallel=True)
# @njit(float64[:, :, :](int32[:, :, :], float64[:], int32))
@njit(float64[:, :, :](typeof(np.array([[[1], [2]], [[3], [4]]])), float64[:], typeof(1)), parallel=True, fastmath=True)
def inv_transform_sax(dataset_sax, breakpoints_middle_, original_size):
    """Compute time series corresponding to given SAX representations.

    Parameters
    ----------
    dataset_sax : array-like, shape=[n_ts, sz, d]
        A dataset of SAX series.
    breakpoints_middle_ : array-like, ndim=1
    original_size : int

    Returns
    -------
    dataset_out : array-like, shape=[n_ts, original_size, d]
    """
    n_ts, sz, d = dataset_sax.shape
    seg_sz = original_size // sz
    dataset_out = np.zeros((n_ts, original_size, d))

    for i in prange(n_ts):
        for t in range(sz):
            t0 = t * seg_sz
            for di in range(d):
                dataset_out[i, t0 : t0 + seg_sz, di] = breakpoints_middle_[
                    dataset_sax[i, t, di]
                ]
    return dataset_out


"""njit --> Ok
tslearn/piecewise/piecewise.py:660:            X_slopes[:, i_seg, :] = cyslopes(X[:, start:end, :], start)
"""


# @njit(parallel=True)
@njit(float64[:, :](float64[:, :, :], int32), parallel=True, fastmath=True)
def cyslopes(dataset, t0):
    """Compute slopes.

    Parameters
    ----------
    dataset : array-like, shape=[n_ts, sz, d]
    t0 : int

    Returns
    -------
    dataset_out : array-like, shape=[n_ts, d]
    """
    n_ts, sz, d = dataset.shape
    dataset_out = np.empty((n_ts, d))
    vec_t = np.arange(t0, t0 + sz).reshape((-1, 1))
    for i in prange(n_ts):
        for di in range(d):
            with objmode(dataset_out_i_di='float64'):
                dataset_out_i_di = (
                    LinearRegression()
                    .fit(vec_t, dataset[i, :, di].reshape((-1, 1)))
                    .coef_[0]
                )
            dataset_out[i, di] = dataset_out_i_di
    return dataset_out


"""njit --> Ok

tslearn/piecewise/piecewise.py:726:        return cydist_1d_sax(sax1, sax2, self.breakpoints_avg_middle_,
"""


# @njit(parallel=True)
@njit(float64(typeof(np.array([[1], [2]])), typeof(np.array([[1], [2]])), float64[:], float64[:], typeof(1)), parallel=True, fastmath=True)
def cydist_1d_sax(
    sax1, sax2, breakpoints_avg_middle_, breakpoints_slope_middle_, original_size
):
    """Compute distance between 1d-SAX representations as defined in [1]_.

    Parameters
    ----------
    sax1 : array-like, shape=[sz, 2 * d]
        1d-SAX representation of a time series.
    sax2 : array-like, shape=[sz, 2 * d]
        1d-SAX representation of another time series.
    breakpoints_avg_middle_ : array-like, ndim=1
    breakpoints_slope_middle_ : array-like, ndim=1
    original_size : int

    Returns
    -------
    dist_1d_sax : float

    Notes
    -----
    Unlike SAX distance, 1d-SAX distance does not lower bound Euclidean
    distance between original time series.

    References
    ----------
    .. [1] S. Malinowski, T. Guyet, R. Quiniou, R. Tavenard. 1d-SAX: a
       Novel Symbolic Representation for Time Series. IDA 2013.
    """
    sz, d_1d_sax = sax1.shape
    assert sz == sax2.shape[0] and d_1d_sax == sax2.shape[1]
    s = 0.0
    d = d_1d_sax // 2
    seg_sz = original_size // sz

    for t in prange(sz):
        t0 = t * seg_sz
        t_middle = float(t0) + 0.5 * seg_sz
        for di in range(d):
            avg1 = breakpoints_avg_middle_[sax1[t, di]]
            avg2 = breakpoints_avg_middle_[sax2[t, di]]
            slope1 = breakpoints_slope_middle_[sax1[t, di + d]]
            slope2 = breakpoints_slope_middle_[sax2[t, di + d]]
            for tt in range(t0, seg_sz * (t + 1)):
                s += (
                    avg1 + slope1 * (tt - t_middle) - (avg2 + slope2 * (tt - t_middle))
                ) ** 2
    dist_1d_sax = np.sqrt(s)
    return dist_1d_sax


"""njit --> Ok

tslearn/piecewise/piecewise.py:771:        X_orig = inv_transform_1d_sax(
"""


# @njit(parallel=True)
@njit(float64[:, :, :](typeof(np.array([[[1], [2]], [[3], [4]]])), float64[:], float64[:], typeof(1)), parallel=True, fastmath=True)
def inv_transform_1d_sax(
    dataset_sax, breakpoints_avg_middle_, breakpoints_slope_middle_, original_size
):
    """Compute time series corresponding to given 1d-SAX representations.

    Parameters
    ----------
    dataset_sax : array-like, shape=[n_ts, sz, 2 * d]
        A dataset of SAX series.
    breakpoints_avg_middle_ : array-like, ndim=1
    breakpoints_slope_middle_ : array-like, ndim=1
    original_size : int

    Returns
    -------
    dataset_out : array-like, shape=[n_ts, original_size, d]
        A dataset of time series corresponding to the provided
            representation.
    """
    n_ts, sz, d_1d_sax = dataset_sax.shape
    d = d_1d_sax // 2
    seg_sz = original_size // sz
    dataset_out = np.empty((n_ts, original_size, d))

    for i in prange(n_ts):
        for t in range(sz):
            t0 = t * seg_sz
            t_middle = float(t0) + 0.5 * (seg_sz - 1)
            for di in range(d):
                avg = breakpoints_avg_middle_[dataset_sax[i, t, di]]
                slope = breakpoints_slope_middle_[dataset_sax[i, t, di + d]]
                for tt in range(t0, seg_sz * (t + 1)):
                    dataset_out[i, tt, di] = avg + slope * (tt - t_middle)
    return dataset_out
