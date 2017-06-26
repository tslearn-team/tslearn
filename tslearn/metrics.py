"""
The :mod:`tslearn.metrics` module gathers time series similarity metrics.
"""

import numpy
from scipy.spatial.distance import pdist
from sklearn.utils import check_random_state
from tslearn.soft_dtw_fast import _soft_dtw, _soft_dtw_grad, _jacobian_product_sq_euc
from sklearn.metrics.pairwise import euclidean_distances

from tslearn.cydtw import dtw as cydtw, dtw_path as cydtw_path, cdist_dtw as cycdist_dtw, lb_envelope as cylb_envelope
from tslearn.cylrdtw import lr_dtw as cylr_dtw, lr_dtw_backtrace as cylr_dtw_backtrace, cdist_lr_dtw as cycdist_lr_dtw
from tslearn.cygak import cdist_gak as cycdist_gak, normalized_gak as cynormalized_gak
from tslearn.utils import to_time_series, to_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def dtw_path(s1, s2):
    """Compute Dynamic Time Warping (DTW) similarity measure between (possibly multidimensional) time series and
    return both the path and the similarity.

    It is not required that both time series share the same size, but they must be the same dimension.
    DTW was originally presented in [1]_.

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
        Matching path represented as a list of index pairs. In each pair, the first index corresponds to s1 and the
        second one corresponds to s2
    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_path([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> dist
    0.0

    See Also
    --------
    dtw : Get only the similarity score for DTW
    cdist_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition,"
       IEEE Transactions on Acoustics, Speech and Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    return cydtw_path(s1, s2)


def dtw(s1, s2):
    """Compute Dynamic Time Warping (DTW) similarity measure between (possibly multidimensional) time series and
    return it.

    It is not required that both time series share the same size, but they must be the same dimension.
    DTW was originally presented in [1]_.

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

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition,"
       IEEE Transactions on Acoustics, Speech and Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    return cydtw(s1, s2)


def cdist_dtw(dataset1, dataset2=None):
    """Compute cross-similarity matrix using Dynamic Time Warping (DTW) similarity measure.

    DTW was originally presented in [1]_.

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

    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for spoken word recognition,"
       IEEE Transactions on Acoustics, Speech and Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """
    dataset1 = to_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = to_time_series_dataset(dataset2)
    return cycdist_dtw(dataset1, dataset2, self_similarity=self_similarity)


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
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    return cylr_dtw(s1, s2, gamma=gamma)[0]


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
    dataset1 = to_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = to_time_series_dataset(dataset2)
    return cycdist_lr_dtw(dataset1, dataset2, gamma=gamma, self_similarity=self_similarity)


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
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    sim, probas = cylr_dtw(s1, s2, gamma=gamma)
    path = cylr_dtw_backtrace(probas)
    return path, sim


def gak(s1, s2, sigma=1.):
    """Compute Global Alignment Kernel (GAK) between (possibly multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must be the same dimension. GAK was
    originally presented in [1]_.

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
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    return cynormalized_gak(s1, s2, sigma)


def cdist_gak(dataset1, dataset2=None, sigma=1.):
    """Compute cross-similarity matrix using Global Alignment kernel (GAK).

    GAK was originally presented in [1]_.

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
    >>> cdist_gak([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 1. , 0.656...],
           [ 0.656..., 1. ]])

    See Also
    --------
    gak : Compute Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    dataset1 = to_time_series_dataset(dataset1)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = to_time_series_dataset(dataset2)
    return cycdist_gak(dataset1, dataset2, sigma, self_similarity=self_similarity)


def sigma_gak(dataset, n_samples=100, random_state=None):
    """Compute sigma value to be used for GAK.

    This method was originally presented in [1]_.

    Parameters
    ----------
    dataset
        A dataset of time series
    n_samples : int (default: 100)
        Number of samples on which median distance should be estimated
    random_state : integer or numpy.RandomState or None (default: None)
        The generator used to draw the samples. If an integer is given, it fixes the seed. Defaults to the global
        numpy random number generator.

    Returns
    -------
    float
        Suggested bandwidth (:math:`\\sigma`) for the Global Alignment kernel

    Example
    -------
    >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
    >>> sigma_gak(dataset=dataset, n_samples=200, random_state=0)  # doctest: +ELLIPSIS
    2.0...

    See Also
    --------
    gak : Compute Global Alignment kernel
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    random_state = check_random_state(random_state)
    dataset = to_time_series_dataset(dataset)
    n_ts, sz, d = dataset.shape
    if n_ts * sz < n_samples:
        replace = True
    else:
        replace = False
    sample_indices = random_state.choice(n_ts * sz, size=n_samples, replace=replace)
    dists = pdist(dataset.reshape((-1, d))[sample_indices], metric="euclidean")
    return numpy.median(dists) * numpy.sqrt(sz)


def gamma_soft_dtw(dataset, n_samples=100, random_state=None):
    """Compute gamma value to be used for GAK/Soft-DTW.

    This method was originally presented in [1]_.

    Parameters
    ----------
    dataset
        A dataset of time series
    n_samples : int (default: 100)
        Number of samples on which median distance should be estimated
    random_state : integer or numpy.RandomState or None (default: None)
        The generator used to draw the samples. If an integer is given, it fixes the seed. Defaults to the global
        numpy random number generator.

    Returns
    -------
    float
        Suggested :math:`\\gamma` parameter for the Soft-DTW

    Example
    -------
    >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
    >>> gamma_soft_dtw(dataset=dataset, n_samples=200, random_state=0)  # doctest: +ELLIPSIS
    8.0...

    See Also
    --------
    sigma_gak : Compute sigma parameter for Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    return 2. * sigma_gak(dataset=dataset, n_samples=n_samples, random_state=random_state) ** 2


def lb_keogh(ts_query, ts_candidate=None, radius=1, envelope_candidate=None):
    """Compute LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts_query : array-like
        Query time-series to compare to the envelope of the candidate.
    ts_candidate : array-like or None (default: None)
        Candidate time-series. None means the envelope is provided via `envelope_query` parameter and hence does not
        need to be computed again.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time index i will be generated based on
        all observations from the candidate time series at indices comprised between i-radius and i+radius). Not used
        if `ts_candidate` is None.
    envelope_candidate: pair of array-like (envelope_down, envelope_up) or None (default: None)
        Pre-computed envelope of the candidate time series. If set to None, it is computed based on `ts_candidate`.

    Returns
    -------
    float
        Distance between the query time series and the envelope of the candidate time series.

    Examples
    --------
    >>> ts1 = [1, 2, 3, 2, 1]
    >>> ts2 = [0, 0, 0, 0, 0]
    >>> env_low, env_up = lb_envelope(ts1, radius=1)
    >>> lb_keogh(ts_query=ts2, envelope_candidate=(env_low, env_up))  # doctest: +ELLIPSIS
    2.8284...
    >>> lb_keogh(ts_query=ts2, ts_candidate=ts1, radius=1)  # doctest: +ELLIPSIS
    2.8284...

    See also
    --------
    lb_envelope : Compute LB_Keogh-related envelope

    References
    ----------
    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International Conference on Very Large Data Bases, 2002.
       pp 406-417.
    """
    if ts_candidate is None:
        envelope_down, envelope_up = envelope_candidate
    else:
        ts_candidate = to_time_series(ts_candidate)
        assert ts_candidate.shape[1] == 1, "LB_Keogh is available only for monodimensional time series"
        envelope_down, envelope_up = lb_envelope(ts_candidate, radius)
    ts_query = to_time_series(ts_query)
    assert ts_query.shape[1] == 1, "LB_Keogh is available only for monodimensional time series"
    indices_up = ts_query[:, 0] > envelope_up[:, 0]
    indices_down = ts_query[:, 0] < envelope_down[:, 0]
    return numpy.sqrt(numpy.linalg.norm(ts_query[indices_up, 0] - envelope_up[indices_up, 0]) ** 2 + \
                      numpy.linalg.norm(ts_query[indices_down, 0] - envelope_down[indices_down, 0]) ** 2)


def lb_envelope(ts, radius=1):
    """Compute time-series envelope as required by LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts : array-like
        Time-series for which the envelope should be computed.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time index i will be generated based on
        all observations from the time series at indices comprised between i-radius and i+radius).

    Returns
    -------
    array-like
        Lower-side of the envelope.
    array-like
        Upper-side of the envelope.

    Examples
    --------
    >>> ts1 = [1, 2, 3, 2, 1]
    >>> env_low, env_up = lb_envelope(ts1, radius=1)
    >>> env_low
    array([[ 1.],
           [ 1.],
           [ 2.],
           [ 1.],
           [ 1.]])
    >>> env_up
    array([[ 2.],
           [ 3.],
           [ 3.],
           [ 3.],
           [ 2.]])

    See also
    --------
    lb_keogh : Compute LB_Keogh similarity

    References
    ----------
    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International Conference on Very Large Data Bases, 2002.
       pp 406-417.
    """
    return cylb_envelope(to_time_series(ts), radius=radius)


def soft_dtw(ts1, ts2, gamma=1.):
    """Compute Soft-DTW metric between two time series.

    Soft-DTW was originally presented in [1]_.

    Parameters
    ----------
    s1
        A time series
    s2
        Another time series
    gamma : float (default 1.)
        Gamma paraneter for Soft-DTW

    Returns
    -------
    float
        Similarity

    Examples
    --------
    >>> soft_dtw([1, 2, 2, 3], [1., 2., 3., 4.], gamma=1.)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    -0.89...
    >>> soft_dtw([1, 2, 3, 3], [1., 2., 2.1, 3.2], gamma=0.01)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    0.089...

    See Also
    --------
    cdist_soft_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for Time-Series," ICML 2017.
    """
    if gamma == 0.:
        return dtw(ts1, ts2)
    return SoftDTW(SquaredEuclidean(ts1, ts2), gamma=numpy.float64(gamma)).compute()


def cdist_soft_dtw(dataset1, dataset2=None, gamma=1.):
    """Compute cross-similarity matrix using Soft-DTW metric.

    Soft-DTW was originally presented in [1]_.

    Parameters
    ----------
    dataset1
        A dataset of time series
    dataset2
        Another dataset of time series
    gamma : float (default 1.)
        Gamma paraneter for Soft-DTW

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], gamma=.01)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[-0.01...,  1. ],
           [ 1.     ,  0. ]])
    >>> cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 2, 3], [1., 2., 3., 4.]], gamma=.01)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[-0.01...,  1. ],
           [ 1.     ,  0. ]])

    See Also
    --------
    soft_dtw : Compute Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for Time-Series," ICML 2017.
    """
    dataset1 = to_time_series_dataset(dataset1, dtype=numpy.float64)
    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = to_time_series_dataset(dataset2, dtype=numpy.float64)
    dists = numpy.empty((dataset1.shape[0], dataset2.shape[0]))
    for i, ts1 in enumerate(dataset1):
        for j, ts2 in enumerate(dataset2):
            if self_similarity and j < i:
                dists[i, j] = dists[j, i]
            else:
                dists[i, j] = soft_dtw(ts1, ts2, gamma=gamma)

    return dists


class SoftDTW(object):
    def __init__(self, D, gamma=1.):
        """
        Parameters
        ----------
        gamma: float
            Regularization parameter.
            Lower is less smoothed (closer to true DTW).

        Attributes
        ----------
        self.R_: array, shape = [m + 2, n + 2]
            Accumulated cost matrix (stored after calling `compute`).
        """
        if hasattr(D, "compute"):
            self.D = D.compute()
        else:
            self.D = D
        self.D = self.D.astype(numpy.float64)

        self.gamma = numpy.float64(gamma)

    def compute(self):
        """
        Compute soft-DTW by dynamic programming.
        Returns
        -------
        sdtw: float
            soft-DTW discrepancy.
        """
        m, n = self.D.shape

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the backward recursion.
        self.R_ = numpy.zeros((m+2, n+2), dtype=numpy.float64)

        _soft_dtw(self.D, self.R_, gamma=self.gamma)

        return self.R_[m, n]

    def grad(self):
        """
        Compute gradient of soft-DTW w.r.t. D by dynamic programming.
        Returns
        -------
        grad: array, shape = [m, n]
            Gradient w.r.t. D.
        """
        if not hasattr(self, "R_"):
            raise ValueError("Needs to call compute() first.")

        m, n = self.D.shape

        # Add an extra row and an extra column to D.
        # Needed to deal with edge cases in the recursion.
        D = numpy.vstack((self.D, numpy.zeros(n)))
        D = numpy.hstack((D, numpy.zeros((m+1, 1))))

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the recursion.
        E = numpy.zeros((m+2, n+2), dtype=numpy.float64)

        _soft_dtw_grad(D, self.R_, E, gamma=self.gamma)

        return E[1:-1, 1:-1]


class SquaredEuclidean(object):

    def __init__(self, X, Y):
        """
        Parameters
        ----------
        X: array, shape = [m, d]
            First time series.
        Y: array, shape = [n, d]
            Second time series.

        Examples
        --------
        >>> SquaredEuclidean([1, 2, 2, 3], [1, 2, 3, 4]).compute()
        array([[ 0.,  1.,  4.,  9.],
               [ 1.,  0.,  1.,  4.],
               [ 1.,  0.,  1.,  4.],
               [ 4.,  1.,  0.,  1.]])
        """
        self.X = to_time_series(X).astype(numpy.float64)
        self.Y = to_time_series(Y).astype(numpy.float64)

    def compute(self):
        """
        Compute distance matrix.
        Returns
        -------
        D: array, shape = [m, n]
            Distance matrix.
        """
        return euclidean_distances(self.X, self.Y, squared=True)

    def jacobian_product(self, E):
        """
        Compute the product between the Jacobian
        (a linear map from m x d to m x n) and a matrix E.
        Parameters
        ----------
        E: array, shape = [m, n]
            Second time series.
        Returns
        -------
        G: array, shape = [m, d]
            Product with Jacobian
            ([m x d, m x n] * [m x n] = [m x d]).
        """
        G = numpy.zeros_like(self.X, dtype=numpy.float64)

        _jacobian_product_sq_euc(self.X, self.Y, E.astype(numpy.float64), G)

        return G
