import numpy
from joblib import Parallel, delayed
from numba import njit
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state

from tslearn.utils import to_time_series, to_time_series_dataset, ts_size, \
    check_equal_size
from .utils import _cdist_generic
from .dtw_variants import dtw, dtw_path
from .soft_dtw_fast import _soft_dtw, _soft_dtw_grad, \
    _jacobian_product_sq_euc

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}
TSLEARN_VALID_METRICS = ["dtw", "gak", "softdtw", "sax"]
VARIABLE_LENGTH_METRICS = ["dtw", "gak", "softdtw", "sax"]


@njit(nogil=True)
def njit_gak(s1, s2, gram):
    l1 = s1.shape[0]
    l2 = s2.shape[0]

    cum_sum = numpy.zeros((l1 + 1, l2 + 1))
    cum_sum[0, 0] = 1.

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = (cum_sum[i, j + 1] +
                                     cum_sum[i + 1, j] +
                                     cum_sum[i, j]) * gram[i, j]

    return cum_sum[l1, l2]


def _gak_gram(s1, s2, sigma=1.):
    gram = - cdist(s1, s2, "sqeuclidean") / (2 * sigma ** 2)
    gram -= numpy.log(2 - numpy.exp(gram))
    return numpy.exp(gram)


def unnormalized_gak(s1, s2, sigma=1.):
    r"""Compute Global Alignment Kernel (GAK) between (possibly
    multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must
    be the same dimension. GAK was
    originally presented in [1]_.
    This is an unnormalized version.

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
    >>> unnormalized_gak([1, 2, 3],
    ...                  [1., 2., 2., 3.],
    ...                  sigma=2.)  # doctest: +ELLIPSIS
    15.358...
    >>> unnormalized_gak([1, 2, 3],
    ...                  [1., 2., 2., 3., 4.])  # doctest: +ELLIPSIS
    3.166...

    See Also
    --------
    gak : normalized version of GAK that ensures that k(x,x) = 1
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    gram = _gak_gram(s1, s2, sigma=sigma)

    gak_val = njit_gak(s1, s2, gram)
    return gak_val


def gak(s1, s2, sigma=1.):  # TODO: better doc (formula for the kernel)
    r"""Compute Global Alignment Kernel (GAK) between (possibly
    multidimensional) time series and return it.

    It is not required that both time series share the same size, but they must
    be the same dimension. GAK was
    originally presented in [1]_.
    This is a normalized version that ensures that :math:`k(x,x)=1` for all
    :math:`x` and :math:`k(x,y) \in [0, 1]` for all :math:`x, y`.

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
    denom = numpy.sqrt(unnormalized_gak(s1, s1, sigma=sigma) *
                       unnormalized_gak(s2, s2, sigma=sigma))
    return unnormalized_gak(s1, s2, sigma=sigma) / denom


def cdist_gak(dataset1, dataset2=None, sigma=1., n_jobs=None, verbose=0):
    r"""Compute cross-similarity matrix using Global Alignment kernel (GAK).

    GAK was originally presented in [1]_.

    Parameters
    ----------
    dataset1
        A dataset of time series
    dataset2
        Another dataset of time series
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> cdist_gak([[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)
    array([[1.        , 0.65629661],
           [0.65629661, 1.        ]])
    >>> cdist_gak([[1, 2, 2], [1., 2., 3., 4.]],
    ...           [[1, 2, 2, 3], [1., 2., 3., 4.], [1, 2, 2, 3]],
    ...           sigma=2.)
    array([[0.71059484, 0.29722877, 0.71059484],
           [0.65629661, 1.        , 0.65629661]])

    See Also
    --------
    gak : Compute Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """  # noqa: E501
    unnormalized_matrix = _cdist_generic(dist_fun=unnormalized_gak,
                                         dataset1=dataset1,
                                         dataset2=dataset2,
                                         n_jobs=n_jobs,
                                         verbose=verbose,
                                         sigma=sigma,
                                         compute_diagonal=True)
    dataset1 = to_time_series_dataset(dataset1)
    if dataset2 is None:
        diagonal = numpy.diag(numpy.sqrt(1. / numpy.diag(unnormalized_matrix)))
        diagonal_left = diagonal_right = diagonal
    else:
        dataset2 = to_time_series_dataset(dataset2)
        diagonal_left = Parallel(n_jobs=n_jobs,
                                 prefer="threads",
                                 verbose=verbose)(
            delayed(unnormalized_gak)(dataset1[i], dataset1[i], sigma=sigma)
            for i in range(len(dataset1))
        )
        diagonal_right = Parallel(n_jobs=n_jobs,
                                  prefer="threads",
                                  verbose=verbose)(
            delayed(unnormalized_gak)(dataset2[j], dataset2[j], sigma=sigma)
            for j in range(len(dataset2))
        )
        diagonal_left = numpy.diag(1. / numpy.sqrt(diagonal_left))
        diagonal_right = numpy.diag(1. / numpy.sqrt(diagonal_right))
    return (diagonal_left.dot(unnormalized_matrix)).dot(diagonal_right)


def sigma_gak(dataset, n_samples=100, random_state=None):
    r"""Compute sigma value to be used for GAK.

    This method was originally presented in [1]_.

    Parameters
    ----------
    dataset
        A dataset of time series
    n_samples : int (default: 100)
        Number of samples on which median distance should be estimated
    random_state : integer or numpy.RandomState or None (default: None)
        The generator used to draw the samples. If an integer is given, it
        fixes the seed. Defaults to the global numpy random number generator.

    Returns
    -------
    float
        Suggested bandwidth (:math:`\sigma`) for the Global Alignment kernel

    Examples
    --------
    >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
    >>> sigma_gak(dataset=dataset,
    ...           n_samples=200,
    ...           random_state=0)  # doctest: +ELLIPSIS
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
    sample_indices = random_state.choice(n_ts * sz,
                                         size=n_samples,
                                         replace=replace)
    dists = pdist(dataset[:, :sz, :].reshape((-1, d))[sample_indices],
                  metric="euclidean")
    return numpy.median(dists) * numpy.sqrt(sz)


def gamma_soft_dtw(dataset, n_samples=100, random_state=None):
    r"""Compute gamma value to be used for GAK/Soft-DTW.

    This method was originally presented in [1]_.

    Parameters
    ----------
    dataset
        A dataset of time series
    n_samples : int (default: 100)
        Number of samples on which median distance should be estimated
    random_state : integer or numpy.RandomState or None (default: None)
        The generator used to draw the samples. If an integer is given, it
        fixes the seed. Defaults to the global numpy random number generator.

    Returns
    -------
    float
        Suggested :math:`\gamma` parameter for the Soft-DTW

    Examples
    --------
    >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
    >>> gamma_soft_dtw(dataset=dataset,
    ...                n_samples=200,
    ...                random_state=0)  # doctest: +ELLIPSIS
    8.0...

    See Also
    --------
    sigma_gak : Compute sigma parameter for Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels," ICML 2011.
    """
    return 2. * sigma_gak(dataset=dataset,
                          n_samples=n_samples,
                          random_state=random_state) ** 2


def soft_dtw(ts1, ts2, gamma=1.):
    r"""Compute Soft-DTW metric between two time series.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

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
    >>> soft_dtw([1, 2, 2, 3],
    ...          [1., 2., 3., 4.],
    ...          gamma=1.)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    -0.89...
    >>> soft_dtw([1, 2, 3, 3],
    ...          [1., 2., 2.1, 3.2],
    ...          gamma=0.01)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    0.089...

    See Also
    --------
    cdist_soft_dtw : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    if gamma == 0.:
        return dtw(ts1, ts2) ** 2
    return SoftDTW(SquaredEuclidean(ts1[:ts_size(ts1)], ts2[:ts_size(ts2)]),
                   gamma=gamma).compute()


def soft_dtw_alignment(ts1, ts2, gamma=1.):
    r"""Compute Soft-DTW metric between two time series and return both the 
    similarity measure and the alignment matrix.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

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
    numpy.ndarray
        Soft-alignment matrix
    float
        Similarity

    Examples
    --------
    >>> a, dist = soft_dtw_alignment([1, 2, 2, 3],
    ...                              [1., 2., 3., 4.],
    ...                              gamma=1.)  # doctest: +ELLIPSIS
    >>> dist
    -0.89...
    >>> a  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    array([[1.00...e+00, 1.88...e-01, 2.83...e-04, 4.19...e-11],
           [3.40...e-01, 8.17...e-01, 8.87...e-02, 3.94...e-05],
           [5.05...e-02, 7.09...e-01, 5.30...e-01, 6.98...e-03],
           [1.37...e-04, 1.31...e-01, 7.30...e-01, 1.00...e+00]])

    See Also
    --------
    soft_dtw : Returns soft-DTW score alone

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    if gamma == 0.:
        path, dist = dtw_path(ts1, ts2)
        dist_sq = dist ** 2
        a = numpy.zeros((ts_size(ts1), ts_size(ts2)))
        for i, j in path:
            a[i, j] = 1.
    else:
        sdtw = SoftDTW(SquaredEuclidean(ts1[:ts_size(ts1)], ts2[:ts_size(ts2)]),
                       gamma=gamma)
        dist_sq = sdtw.compute()
        a = sdtw.grad()
    return a, dist_sq


def cdist_soft_dtw(dataset1, dataset2=None, gamma=1.):
    r"""Compute cross-similarity matrix using Soft-DTW metric.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

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
    >>> cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], gamma=.01)
    array([[-0.01098612,  1.        ],
           [ 1.        ,  0.        ]])
    >>> cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]],
    ...                [[1, 2, 2, 3], [1., 2., 3., 4.]], gamma=.01)
    array([[-0.01098612,  1.        ],
           [ 1.        ,  0.        ]])

    See Also
    --------
    soft_dtw : Compute Soft-DTW
    cdist_soft_dtw_normalized : Cross similarity matrix between time series
        datasets using a normalized version of Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
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


def cdist_soft_dtw_normalized(dataset1, dataset2=None, gamma=1.):
    r"""Compute cross-similarity matrix using a normalized version of the
    Soft-DTW metric.

    Soft-DTW was originally presented in [1]_ and is
    discussed in more details in our
    :ref:`user-guide page on DTW and its variants<dtw>`.

    Soft-DTW is computed as:

    .. math::

        \text{soft-DTW}_{\gamma}(X, Y) =
            \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} \|X_i, Y_j\|^2

    where :math:`\min^\gamma` is the soft-min operator of parameter
    :math:`\gamma`.

    In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
    hard-min operator and soft-DTW is defined as the square of the DTW
    similarity measure.

    This normalized version is defined as:

    .. math::

        \text{norm-soft-DTW}_{\gamma}(X, Y) =
            \text{soft-DTW}_{\gamma}(X, Y) -
            \frac{1}{2} \left(\text{soft-DTW}_{\gamma}(X, X) +
                \text{soft-DTW}_{\gamma}(Y, Y)\right)

    and ensures that all returned values are positive and that
    :math:`\text{norm-soft-DTW}_{\gamma}(X, X) = 0`.

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
    >>> time_series = numpy.random.randn(10, 15, 1)
    >>> numpy.alltrue(cdist_soft_dtw_normalized(time_series) >= 0.)
    True

    See Also
    --------
    soft_dtw : Compute Soft-DTW
    cdist_soft_dtw : Cross similarity matrix between time series
        datasets using the unnormalized version of Soft-DTW

    References
    ----------
    .. [1] M. Cuturi, M. Blondel "Soft-DTW: a Differentiable Loss Function for
       Time-Series," ICML 2017.
    """
    dists = cdist_soft_dtw(dataset1, dataset2=dataset2, gamma=gamma)
    d_ii = numpy.diag(dists)
    dists -= .5 * (d_ii.reshape((-1, 1)) + d_ii.reshape((1, -1)))
    return dists


class SoftDTW:
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
        self.R_ = numpy.zeros((m + 2, n + 2), dtype=numpy.float64)
        self.computed = False

        self.gamma = numpy.float64(gamma)

    def compute(self):
        """Compute soft-DTW by dynamic programming.

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
        """Compute gradient of soft-DTW w.r.t. D by dynamic programming.

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
        D = numpy.hstack((D, numpy.zeros((m + 1, 1))))

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the recursion.
        E = numpy.zeros((m + 2, n + 2), dtype=numpy.float64)

        _soft_dtw_grad(D, self.R_, E, gamma=self.gamma)

        return E[1:-1, 1:-1]


class SquaredEuclidean:
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
        array([[0., 1., 4., 9.],
               [1., 0., 1., 4.],
               [1., 0., 1., 4.],
               [4., 1., 0., 1.]])
        """
        self.X = to_time_series(X).astype(numpy.float64)
        self.Y = to_time_series(Y).astype(numpy.float64)

    def compute(self):
        """Compute distance matrix.

        Returns
        -------
        D: array, shape = [m, n]
            Distance matrix.
        """
        return euclidean_distances(self.X, self.Y, squared=True)

    def jacobian_product(self, E):
        """Compute the product between the Jacobian
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
