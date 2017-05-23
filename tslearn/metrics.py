import numpy
import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    import pyximport; pyximport.install()

from tslearn import cydtw
from tslearn import cylrdtw
from tslearn import cygak
from tslearn.utils import npy2d_time_series, npy3d_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def dtw_path(s1, s2):
    """Compute Dynamic Time Warping (DTW) [1]_ similarity measure between (possibly multidimensional) time series and
    return both the path and the similarity.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1
        A time series
    s2
        Another dataset of time series
        If not given, self-similarity of dataset1 is returned

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
    cdist_dtw : Cross similarity matrix between time series datasets
    lr_dtw : Locally_regularized Dynamic Time Warping (LR-DTW) score
    lr_dtw_path : Get both the matching path and the similarity score for LR-DTW

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition,"
       IEEE Transactions on Acoustics, Speech and Signal Processing, vol. 26(1), pp. 43–49, 1978.
    """
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cydtw.dtw_path(s1, s2)


def dtw(s1, s2):
    """Compute Dynamic Time Warping (DTW) [1]_ similarity measure between (possibly multidimensional) time series and
    return it.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1
        A time series
    s2
        Another time series

    Returns
    -------
    float
        Similarity score

    Examples
    --------
    >>> dtw([1, 2, 3], [1., 2., 2., 3.])
    0.0
    >>> dtw([1, 2, 3], [1., 2., 2., 3., 4.])
    1.0

    See Also
    --------
    dtw_path : Get both the matching path and the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets
    lr_dtw : Locally_regularized Dynamic Time Warping (LR-DTW) score
    lr_dtw_path : Get both the matching path and the similarity score for LR-DTW
    """
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cydtw.dtw(s1, s2)


def cdist_dtw(dataset1, dataset2=None):
    """Compute cross-similarity matrix using Dynamic Time Warping (DTW) [1]_ similarity measure.

    Parameters
    ----------
    dataset1
        A dataset of time series
    dataset2
        Another time series

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]])  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0., 1.],
           [ 1., 0.]])
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 3], [2, 3, 4]])  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. ,  1.41421356],
           [ 1. ,  1. ]])

    See Also
    --------
    dtw : Get DTW similarity score
    """
    dataset1 = npy3d_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = npy3d_time_series_dataset(dataset2)
    return cydtw.cdist_dtw(dataset1, dataset2, self_similarity=self_similarity)


def lr_dtw(s1, s2, gamma=0.):
    """Compute Locally Regularized DTW (LR-DTW) similarity measure between (possibly multidimensional) time series and
    return it.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1
        A time series
    s2
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
    s1
        A time series
    s2
        Another time series
    gamma : float
        Regularization parameter

    Returns
    -------
    numpy.ndarray of shape (s1.shape[0], s2.shape[0])
        Matching path represented as a probability map
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


def gak(s1, s2, sigma=1.):
    """Compute Global Alignment Kernel) [2]_ between (possibly multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1
        A time series
    s2
        Another time series
    sigma : float
        Bandwidth of the internal gaussian kernel used for GAK

    Returns
    -------
    float
        Kernel value

    Examples
    --------
    >>> gak([1, 2, 3], [1., 2., 2., 3.], sigma=2.)
    0.83933690793680882
    >>> gak([1, 2, 3], [1., 2., 2., 3., 4.])
    0.2733855908733378

    See Also
    --------
    cdist_gak : Get both the matching path and the similarity score for LR-DTW

    References
    ----------
    .. [2] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cygak.normalized_gak(s1, s2, sigma)


def cdist_gak(dataset1, dataset2=None, sigma=1.):
    """Compute cross-similarity matrix using Global Alignment kernel (GAK) [2]_.

    Parameters
    ----------
    dataset1
        A dataset of time series
    dataset2
        Another time series
    sigma : float
        Bandwidth of the internal gaussian kernel used for GAK

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_gak([[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. , 0.65629661],
           [ 0.65629661, 0. ]])

    See Also
    --------
    gak : Compute Global Alignment Kernel
    """
    dataset1 = npy3d_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = npy3d_time_series_dataset(dataset2)
    return cygak.cdist_gak(dataset1, dataset2, sigma, self_similarity=self_similarity)
