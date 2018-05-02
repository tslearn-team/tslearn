"""
The :mod:`tslearn.metrics` module gathers time series similarity metrics.
"""

import numpy
from scipy.spatial.distance import pdist
from sklearn.utils import check_random_state
from tslearn.soft_dtw_fast import _soft_dtw, _soft_dtw_grad, _jacobian_product_sq_euc
from sklearn.metrics.pairwise import euclidean_distances

from tslearn.cydtw import dtw as cydtw, dtw_path as cydtw_path, cdist_dtw as cycdist_dtw, \
    dtw_subsequence_path as cydtw_subsequence_path
from tslearn.cydtw import lb_envelope as cylb_envelope
from tslearn.cydtw import sakoe_chiba_mask as cysakoe_chiba_mask, itakura_mask as cyitakura_mask
from tslearn.cygak import cdist_gak as cycdist_gak, cdist_normalized_gak as cycdist_normalized_gak, \
    normalized_gak as cynormalized_gak, gak as cygak
from tslearn.utils import to_time_series, to_time_series_dataset, ts_size, check_equal_size

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def dtw_path(s1, s2, global_constraint=None, sakoe_chiba_radius=1):
    """Compute Dynamic Time Warping (DTW) similarity measure between (possibly multidimensional) time series and
    return both the path and the similarity.

    It is not required that both time series share the same size, but they must be the same dimension.
    DTW was originally presented in [1]_.

    Parameters
    ----------
    s1
        A time series.
    s2
        Another time series.
        If not given, self-similarity of dataset1 is returned.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int (default: 1)
        Radius to be used for Sakoe-Chiba band global constraint. Used only if global_constraint is "sakoe_chiba".

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
    >>> dtw_path([1, 2, 3], [1., 2., 2., 3., 4.])[1]
    1.0

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
    sz1 = s1.shape[0]
    sz2 = s2.shape[0]
    if global_constraint == "sakoe_chiba":
        return cydtw_path(s1, s2, mask=sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius))
    elif global_constraint == "itakura":
        return cydtw_path(s1, s2, mask=itakura_mask(sz1, sz2))
    return cydtw_path(s1, s2, mask=numpy.zeros((sz1, sz2)))


def dtw(s1, s2, global_constraint=None, sakoe_chiba_radius=1):
    """Compute Dynamic Time Warping (DTW) similarity measure between (possibly multidimensional) time series and
    return it.

    It is not required that both time series share the same size, but they must be the same dimension.
    DTW was originally presented in [1]_.

    Parameters
    ----------
    s1
        A time series.
    s2
        Another time series.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int (default: 1)
        Radius to be used for Sakoe-Chiba band global constraint. Used only if global_constraint is "sakoe_chiba".

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
    sz1 = s1.shape[0]
    sz2 = s2.shape[0]
    if global_constraint == "sakoe_chiba":
        return cydtw(s1, s2, mask=sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius))
    elif global_constraint == "itakura":
        return cydtw(s1, s2, mask=itakura_mask(sz1, sz2))
    return cydtw(s1, s2, mask=numpy.zeros((sz1, sz2)))


def dtw_subsequence_path(subseq, longseq):
    """Compute sub-sequence Dynamic Time Warping (DTW) similarity measure between a (possibly multidimensional)
    query and a long time series and return both the path and the similarity.

    It is not required that both time series share the same size, but they must be the same dimension.
    This implementation finds the best matching starting and ending positions for `subseq` inside `longseq`.

    Parameters
    ----------
    subseq
        A query time series.
    longseq
        A reference (supposed to be longer than `subseq`) time series.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the first index corresponds to `subseq` and
        the second one corresponds to `longseq`
    float
        Similarity score

    Examples
    --------
    >>> path, dist = dtw_subsequence_path([2, 3], [1., 2., 2., 3., 4.])
    >>> path
    [(0, 2), (1, 3)]
    >>> dist
    0.0

    See Also
    --------
    dtw : Get the similarity score for DTW
    """
    subseq = to_time_series(subseq)
    longseq = to_time_series(longseq)
    return cydtw_subsequence_path(subseq=subseq, longseq=longseq)


def sakoe_chiba_mask(sz1, sz2, radius=1):
    """
    Examples
    --------
    >>> sakoe_chiba_mask(4, 4, 1)  # doctest: +NORMALIZE_WHITESPACE
    array([[  0.,  0., inf, inf],
           [  0.,  0.,  0., inf],
           [ inf,  0.,  0.,  0.],
           [ inf, inf,  0.,  0.]])
    >>> sakoe_chiba_mask(7, 3, 1)  # doctest: +NORMALIZE_WHITESPACE
    array([[  0., 0., inf],
           [  0., 0., inf],
           [  0., 0., inf],
           [  0., 0.,  0.],
           [ inf, 0.,  0.],
           [ inf, 0.,  0.],
           [ inf, 0.,  0.]])
    """
    return cysakoe_chiba_mask(sz1, sz2, radius)


def itakura_mask(sz1, sz2):
    """
    Examples
    --------
    >>> itakura_mask(6, 6)  # doctest: +NORMALIZE_WHITESPACE
    array([[  0., inf, inf, inf, inf, inf],
           [ inf,  0.,  0., inf, inf, inf],
           [ inf,  0.,  0.,  0., inf, inf],
           [ inf, inf,  0.,  0.,  0., inf],
           [ inf, inf, inf,  0.,  0., inf],
           [ inf, inf, inf, inf, inf,  0.]])
    """
    return cyitakura_mask(sz1, sz2)


def cdist_dtw(dataset1, dataset2=None, global_constraint=None, sakoe_chiba_radius=1):
    """Compute cross-similarity matrix using Dynamic Time Warping (DTW) similarity measure.

    DTW was originally presented in [1]_.

    Parameters
    ----------
    dataset1 : array-like
        A dataset of time series
    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of `dataset1` is returned.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int (default: 1)
        Radius to be used for Sakoe-Chiba band global constraint. Used only if global_constraint is "sakoe_chiba".

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]])  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0., 1.],
           [ 1., 0.]])
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 3], [2, 3, 4, 5]])  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 0. ,  2.449...],
           [ 1. ,  1.414...]])

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
    sz1 = dataset1.shape[1]
    sz2 = dataset2.shape[1]
    if global_constraint == "sakoe_chiba":
        return cycdist_dtw(dataset1, dataset2, self_similarity=self_similarity,
                           mask=sakoe_chiba_mask(sz1, sz2, radius=sakoe_chiba_radius))
    elif global_constraint == "itakura":
        return cycdist_dtw(dataset1, dataset2, self_similarity=self_similarity, mask=itakura_mask(sz1, sz2))
    return cycdist_dtw(dataset1, dataset2, self_similarity=self_similarity, mask=numpy.zeros((sz1, sz2)))


def gak(s1, s2, sigma=1.):
    """Compute Global Alignment Kernel (GAK) between (possibly multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must be the same dimension. GAK was
    originally presented in [1]_.
    This is a normalized version that ensures that $k(x,x)=1$ for all $x$ and $k(x,y) \in [0, 1]$ for all $x, y$.

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
    >>> cdist_gak([[1, 2, 2], [1., 2., 3., 4.]], [[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    array([[ 0.710...,  0.297...],
           [ 0.656...,  1.        ]])

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
    return cycdist_normalized_gak(dataset1, dataset2, sigma, self_similarity=self_similarity)


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
    if not check_equal_size(dataset):
        sz = numpy.min([ts_size(ts) for ts in dataset])
    if n_ts * sz < n_samples:
        replace = True
    else:
        replace = False
    sample_indices = random_state.choice(n_ts * sz, size=n_samples, replace=replace)
    dists = pdist(dataset[:, :sz, :].reshape((-1, d))[sample_indices], metric="euclidean")
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

    Note
    ----
        This method requires a `ts_query` and `ts_candidate` (or `envelope_candidate`, depending on the call)
        to be of equal size.

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
    ts1
        A time series
    ts2
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
    return SoftDTW(SquaredEuclidean(ts1[:ts_size(ts1)], ts2[:ts_size(ts2)]), gamma=gamma).compute()


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
    equal_size_ds1 = check_equal_size(dataset1)
    equal_size_ds2 = check_equal_size(dataset2)
    for i, ts1 in enumerate(dataset1):
        if equal_size_ds1:
            ts1_short = ts1
        else:
            ts1_short = ts1[:ts_size(ts1)]
        for j, ts2 in enumerate(dataset2):
            if equal_size_ds2:
                ts2_short = ts2
            else:
                ts2_short = ts2[:ts_size(ts2)]
            if self_similarity and j < i:
                dists[i, j] = dists[j, i]
            else:
                dists[i, j] = soft_dtw(ts1_short, ts2_short, gamma=gamma)

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

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the backward recursion.
        m, n = self.D.shape
        self.R_ = numpy.zeros((m+2, n+2), dtype=numpy.float64)
        self.computed = False

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

        _soft_dtw(self.D, self.R_, gamma=self.gamma)

        self.computed = True

        return self.R_[m, n]

    def grad(self):
        """
        Compute gradient of soft-DTW w.r.t. D by dynamic programming.
        Returns
        -------
        grad: array, shape = [m, n]
            Gradient w.r.t. D.
        """
        if not self.computed:
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
