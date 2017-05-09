import numpy

from tslearn import cydtw
from tslearn import cylrdtw
from tslearn.utils import npy2d_time_series

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def dtw_path(s1, s2):
    """Compute DTW similarity measure between (possibly multidimensional) time series and return both the path and the
    similarity.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1 : numpy.ndarray
        A time series
    s2 : numpy.ndarray
        Another time series

    Returns
    -------

    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the first index corresponds to ``s1`` and the
        second one corresponds to ``s2``
    float
        Similarity score

    See Also
    --------

    dtw : Get only the similarity score for DTW
    lr_dtw : Locally_regularized Dynamic Time Warping (LR-DTW) score
    lr_dtw_path : Get both the matching path and the similarity score for LR-DTW
    """
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cydtw.dtw_path(s1, s2)


def dtw(s1, s2):
    """Compute DTW similarity measure between (possibly multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1 : numpy.ndarray
        A time series
    s2 : numpy.ndarray
        Another time series

    Returns
    -------

    float
        Similarity score

    See Also
    --------

    dtw_path : Get both the matching path and the similarity score for DTW
    lr_dtw : Locally_regularized Dynamic Time Warping (LR-DTW) score
    lr_dtw_path : Get both the matching path and the similarity score for LR-DTW
    """
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cydtw.dtw(s1, s2)


def lr_dtw(s1, s2, gamma=0.):
    """Compute Locally Regularized DTW (LR-DTW) similarity measure between (possibly multidimensional) time series and
    return it.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1 : numpy.ndarray
        A time series
    s2 : numpy.ndarray
        Another time series
    gamma : float
        Regularization parameter

    Returns
    -------

    float
        Similarity score

    See Also
    --------

    lr_dtw_path : Get both the matching path and the similarity score for LR-DTW
    dtw : Dynamic Time Warping score
    dtw_path : Get both the matching path and the similarity score for DTW
    """
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cylrdtw.lr_dtw(s1, s2, gamma=gamma)[0]


def lr_dtw_path(s1, s2, gamma=0.):
    """Compute Locally Regularized DTW (LR-DTW) similarity measure between (possibly multidimensional) time series and
    return both the (probabilistic) path and the similarity.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1 : numpy.ndarray
        A time series
    s2 : numpy.ndarray
        Another time series
    gamma : float
        Regularization parameter

    Returns
    -------

    float
        Similarity score

    See Also
    --------

    lr_dtw : LR-DTW score
    dtw : Dynamic Time Warping (DTW) score
    dtw_path : Get both the matching path and the similarity score for DTW
    """
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    sim, probas = cylrdtw.lr_dtw(s1, s2, gamma=gamma)
    path = cylrdtw.lr_dtw_backtrace(probas)
    return path, sim
