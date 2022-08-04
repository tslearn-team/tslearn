STUFF_cysax = "cysax"

import numpy as np
from sklearn.linear_model import LinearRegression

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"

DTYPE = np.float
DTYPE_INT = np.int


def inv_transform_paa(dataset_paa, original_size):
    n_ts = dataset_paa.shape[0]
    sz = dataset_paa.shape[1]
    d = dataset_paa.shape[2]
    i = 0
    t = 0
    di = 0
    t0 = 0
    seg_sz = original_size // sz
    dataset_out = np.zeros((n_ts, original_size, d))

    for i in range(n_ts):
        for t in range(sz):
            t0 = t * seg_sz
            for di in range(d):
                dataset_out[i, t0 : t0 + seg_sz, di] = dataset_paa[i, t, di]
    return dataset_out


def cydist_sax(sax1, sax2, breakpoints, original_size):
    assert sax1.shape[0] == sax2.shape[0] and sax1.shape[1] == sax2.shape[1]
    s = 0.0
    sz = sax1.shape[0]
    d = sax1.shape[1]
    t = 0
    di = 0
    for t in range(sz):
        for di in range(d):
            if np.abs(sax1[t, di] - sax2[t, di]) > 1:
                max_symbol = max(sax1[t, di], sax2[t, di])
                min_symbol = min(sax1[t, di], sax2[t, di])
                s += (breakpoints[max_symbol - 1] - breakpoints[min_symbol]) ** 2
    return np.sqrt(s * float(original_size) / sz)


def inv_transform_sax(dataset_sax, breakpoints_middle_, original_size):
    n_ts = dataset_sax.shape[0]
    sz = dataset_sax.shape[1]
    d = dataset_sax.shape[2]
    i = 0
    t = 0
    di = 0
    t0 = 0
    seg_sz = original_size // sz
    dataset_out = np.zeros((n_ts, original_size, d))

    for i in range(n_ts):
        for t in range(sz):
            t0 = t * seg_sz
            for di in range(d):
                dataset_out[i, t0 : t0 + seg_sz, di] = breakpoints_middle_[
                    dataset_sax[i, t, di]
                ]
    return dataset_out


def cyslopes(dataset, t0):
    i = 0
    d = 0
    sz = dataset.shape[1]
    dataset_out = np.empty((dataset.shape[0], dataset.shape[2]))
    vec_t = np.arange(t0, t0 + sz).reshape((-1, 1))

    for i in range(dataset.shape[0]):
        for d in range(dataset.shape[2]):
            dataset_out[i, d] = (
                LinearRegression()
                .fit(vec_t, dataset[i, :, d].reshape((-1, 1)))
                .coef_[0]
            )
    return dataset_out


def cydist_1d_sax(
    sax1, sax2, breakpoints_avg_middle_, breakpoints_slope_middle_, original_size
):
    assert sax1.shape[0] == sax2.shape[0] and sax1.shape[1] == sax2.shape[1]
    s = 0.0
    sz = sax1.shape[0]
    d = sax1.shape[1] // 2
    t = 0
    di = 0
    t0 = 0
    seg_sz = original_size // sz
    t_middle = 0.0
    slope1 = 0.0
    slope2 = 0.0
    avg1 = 0.0
    avg2 = 0.0

    for t in range(sz):
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
    return np.sqrt(s)


def inv_transform_1d_sax(
    dataset_sax, breakpoints_avg_middle_, breakpoints_slope_middle_, original_size
):

    n_ts = dataset_sax.shape[0]
    sz = dataset_sax.shape[1]
    d = dataset_sax.shape[2] // 2
    i = 0
    t = 0
    di = 0
    t0 = 0
    seg_sz = original_size // sz
    t_middle = 0.0
    slope = 0.0
    avg = 0.0
    dataset_out = np.empty((n_ts, original_size, d))

    for i in range(n_ts):
        for t in range(sz):
            t0 = t * seg_sz
            t_middle = float(t0) + 0.5 * (seg_sz - 1)
            for di in range(d):
                avg = breakpoints_avg_middle_[dataset_sax[i, t, di]]
                slope = breakpoints_slope_middle_[dataset_sax[i, t, di + d]]
                for tt in range(t0, seg_sz * (t + 1)):
                    dataset_out[i, tt, di] = avg + slope * (tt - t_middle)
    return dataset_out
