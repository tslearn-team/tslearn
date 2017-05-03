import numpy

import cydtw

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

def npy2d_time_series(ts):
    if ts.ndim == 1:
        ts = ts.reshape((-1, 1))
    if ts.dtype != numpy.float:
        ts = ts.astype(numpy.float)
    return ts


def dtw_path(s1, s2):
    """Compute DTW similarity measure between (possibly multidimensional) time series and return both the path and the similarity.
    Time series must be 2d numpy arrays of shape (size, dim) or 1d arrays of shape (size, ).
    It is not required that both time series share the same size, but they must be the same dimension."""
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cydtw.dtw_path(s1, s2)


def dtw(s1, s2):
    """Compute DTW similarity measure between (possibly multidimensional) time series and return it.
    Time series must be 2d numpy arrays of shape (size, dim) or 1d arrays of shape (size, ).
    It is not required that both time series share the same size, but they must be the same dimension."""
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cydtw.dtw(s1, s2)