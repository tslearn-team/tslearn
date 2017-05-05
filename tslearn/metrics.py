import numpy

from tslearn import cydtw
from tslearn import cylrdtw
from tslearn.utils import npy2d_time_series

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def dtw_path(s1, s2):
    """Compute DTW similarity measure between (possibly multidimensional) time series and return both the path and the
    similarity.
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


def lr_dtw(s1, s2, gamma=0.):
    """Compute Locally Regularized DTW similarity measure between (possibly multidimensional) time series and return it.
    Time series must be 2d numpy arrays of shape (size, dim) or 1d arrays of shape (size, ).
    It is not required that both time series share the same size, but they must be the same dimension."""
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cylrdtw.lr_dtw(s1, s2, gamma=gamma)[0]


def lr_dtw_path(s1, s2, gamma=0.):
    """Compute Locally Regularized DTW similarity measure between (possibly multidimensional) time series and return
    both the (probabilistic) path and the similarity.
    Time series must be 2d numpy arrays of shape (size, dim) or 1d arrays of shape (size, ).
    It is not required that both time series share the same size, but they must be the same dimension."""
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    sim, probas = cylrdtw.lr_dtw(s1, s2, gamma=gamma)
    path = cylrdtw.lr_dtw_backtrace(probas)
    return path, sim
