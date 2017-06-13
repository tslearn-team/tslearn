import numpy
import os
from scipy.spatial.distance import pdist
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
    dataset1 : array-like
        A dataset of time series
    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of `dataset1` is returned.

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
    gamma : float (default: 0.)
        Regularization parameter

    Returns
    -------
    float
        Similarity score

    See Also
    --------
    lr_dtw_path : Get both the matching path and the similarity score for LR-DTW
    cdist_lr_dtw : Cross similarity matrix between time series datasets
    dtw : Dynamic Time Warping score
    dtw_path : Get both the matching path and the similarity score for DTW
    """
    s1 = npy2d_time_series(s1)
    s2 = npy2d_time_series(s2)
    return cylrdtw.lr_dtw(s1, s2, gamma=gamma)[0]


def cdist_lr_dtw(dataset1, dataset2=None, gamma=0.):
    """Compute cross-similarity matrix using Locally-Regularized Dynamic Time Warping (LR-DTW) similarity measure.

    Parameters
    ----------
    dataset1 : array-like
        A dataset of time series
    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of `dataset1` is returned.
    gamma : float (default: 0.)
        :math:`\\gamma` parameter for the LR-DTW metric.

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    See Also
    --------
    lr_dtw : Get LR-DTW similarity score
    """
    dataset1 = npy3d_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = npy3d_time_series_dataset(dataset2)
    return cylrdtw.cdist_lr_dtw(dataset1, dataset2, gamma=gamma, self_similarity=self_similarity)


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
    gamma : float (default: 0.)
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
    """Compute Global Alignment Kernel (GAK) [2]_ between (possibly multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must be the same dimension.

    Parameters
    ----------
    s1
        A time series
    s2
        Another time series
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK

    Returns
    -------
    float
        Kernel value

    Examples
    --------
    >>> gak([1, 2, 3], [1., 2., 2., 3.], sigma=2.)  # doctest: +ELLIPSIS
    0.839...
    >>> gak([1, 2, 3], [1., 2., 2., 3., 4.])  # doctest: +ELLIPSIS
    0.273...

    See Also
    --------
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

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
        Another dataset of time series
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_gak([[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 1. , 0.656...],
           [ 0.656..., 1. ]])

    See Also
    --------
    gak : Compute Global Alignment kernel
    """
    dataset1 = npy3d_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = npy3d_time_series_dataset(dataset2)
    return cygak.cdist_gak(dataset1, dataset2, sigma, self_similarity=self_similarity)


def sigma_gak(dataset, n_samples=100):
    """Compute sigma value to be used for GAK as suggested in [2]_.

    Parameters
    ----------
    dataset
        A dataset of time series
    n_samples : int (default: 100)
        Number of samples on which median distance should be estimated

    Returns
    -------
    float
        Suggested bandwidth (:math:`\\sigma`) for the Global Alignment kernel

    See Also
    --------
    gak : Compute Global Alignment kernel
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel
    """
    dataset = npy3d_time_series_dataset(dataset)
    n_ts, sz, d = dataset.shape
    if n_ts * sz < n_samples:
        replace = True
    else:
        replace = False
    sample_indices = numpy.random.choice(n_ts * sz, size=n_samples, replace=replace)
    dists = pdist(dataset.reshape((-1, d))[sample_indices], metric="euclidean")
    return numpy.median(dists) * numpy.sqrt(sz)


def lb_keogh(ts_query, ts_candidate=None, radius=1, enveloppe_candidate=None):
    """Compute LB_Keogh as defined in [3]_.

    Parameters
    ----------
    ts_query : array-like
        Query time-series to compare to the enveloppe of the candidate.
    ts_candidate : array-like or None (default: None)
        Candidate time-series. None means the enveloppe is provided via `enveloppe_query` parameter and hence does not
        need to be computed again.
    radius : int (default: 1)
        Radius to be used for the enveloppe generation (the enveloppe at time index i will be generated based on
        all observations from the candidate time series at indices comprised between i-radius and i+radius). Not used
        if `ts_candidate` is None.
    enveloppe_candidate: pair of array-like (enveloppe_down, enveloppe_up) or None (default: None)
        Pre-computed enveloppe of the candidate time series. If set to None, it is computed based on `ts_candidate`.

    Returns
    -------
    float
        Distance between the query time series and the enveloppe of the candidate time series.

    References
    ----------
    .. [3] Keogh, E. Exact indexing of dynamic time warping. In International Conference on Very Large Data Bases, 2002.
       pp 406-417.
    """
    if ts_candidate is None:
        enveloppe_down, enveloppe_up = enveloppe_candidate
    else:
        ts_candidate = npy2d_time_series(ts_candidate)
        assert ts_candidate.shape[1] == 1, "LB_Keogh is available only for monodimensional time series"
        enveloppe_down, enveloppe_up = lb_enveloppe(ts_candidate, radius)
    ts_query = npy2d_time_series(ts_query)
    assert ts_query.shape[1] == 1, "LB_Keogh is available only for monodimensional time series"
    indices_up = ts_query[:, 0] > enveloppe_up[:, 0]
    indices_down = ts_query[:, 0] < enveloppe_down[:, 0]
    return numpy.sum((ts_query[indices_up, 0] - enveloppe_up[indices_up, 0]) ** 2) + \
           numpy.sum((ts_query[indices_down, 0] - enveloppe_down[indices_down, 0]) ** 2)


def lb_enveloppe(ts, radius=1):
    """Compute time-series enveloppe as required by LB_Keogh [3]_.

    Parameters
    ----------
    ts : array-like
        Time-series for which the enveloppe should be computed.
    radius : int (default: 1)
        Radius to be used for the enveloppe generation (the enveloppe at time index i will be generated based on
        all observations from the time series at indices comprised between i-radius and i+radius).

    Returns
    -------
    array-like
        Lower-side of the enveloppe.
    array-like
        Upper-side of the enveloppe.
    """
    return cydtw.lb_enveloppe(ts, radius=radius)