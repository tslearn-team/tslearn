import numpy as np
from joblib import Parallel, delayed
from numba import njit
from sklearn.utils import check_random_state

from tslearn.backend import instantiate_backend
from tslearn.utils import (
    check_equal_size,
    to_time_series,
    to_time_series_dataset,
    ts_size,
)

from .dtw_variants import dtw, dtw_path
from .soft_dtw_fast import (
    _jacobian_product_sq_euc,
    _njit_jacobian_product_sq_euc,
    _njit_soft_dtw,
    _njit_soft_dtw_grad,
    _soft_dtw,
    _soft_dtw_grad,
)
from .utils import _cdist_generic

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}
TSLEARN_VALID_METRICS = ["dtw", "gak", "softdtw", "sax"]
VARIABLE_LENGTH_METRICS = ["dtw", "gak", "softdtw", "sax"]


def _gak(s1, s2, gram, be=None):
    be = instantiate_backend(be, s1)
    l1 = s1.shape[0]
    l2 = s2.shape[0]

    cum_sum = be.zeros((l1 + 1, l2 + 1))
    cum_sum[0, 0] = 1.0

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = (
                cum_sum[i, j + 1] + cum_sum[i + 1, j] + cum_sum[i, j]
            ) * gram[i, j]

    return cum_sum[l1, l2]


@njit(nogil=True)
def _njit_gak(s1, s2, gram):
    l1 = s1.shape[0]
    l2 = s2.shape[0]

    cum_sum = np.zeros((l1 + 1, l2 + 1))
    cum_sum[0, 0] = 1.0

    for i in range(l1):
        for j in range(l2):
            cum_sum[i + 1, j + 1] = (
                cum_sum[i, j + 1] + cum_sum[i + 1, j] + cum_sum[i, j]
            ) * gram[i, j]

    return cum_sum[l1, l2]


def _gak_gram(s1, s2, sigma=1.0, be=None):
    be = instantiate_backend(be, s1)
    gram = -be.cdist(s1, s2, "sqeuclidean") / (2 * sigma**2)
    gram = be.array(gram)
    gram -= be.log(2 - be.exp(gram))
    return be.exp(gram)


def unnormalized_gak(s1, s2, sigma=1.0, be=None):
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
    be : Backend object or string or None
        Backend.

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
    be = instantiate_backend(be, s1)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    gram = _gak_gram(s1, s2, sigma=sigma, be=be)

    if be.is_numpy:
        return _njit_gak(s1, s2, gram)
    return _gak(s1, s2, gram, be=be)


def gak(s1, s2, sigma=1.0, be=None):  # TODO: better doc (formula for the kernel)
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
    be : Backend object or string or None
        Backend.

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
    be = instantiate_backend(be, s1)
    denom = be.sqrt(
        unnormalized_gak(s1, s1, sigma=sigma, be=be)
        * unnormalized_gak(s2, s2, sigma=sigma, be=be)
    )
    return unnormalized_gak(s1, s2, sigma=sigma, be=be) / denom


def cdist_gak(dataset1, dataset2=None, sigma=1.0, n_jobs=None, verbose=0, be=None):
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
    be : Backend object or string or None
        Backend.

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
    be = instantiate_backend(be, dataset1)

    unnormalized_matrix = _cdist_generic(
        dist_fun=unnormalized_gak,
        dataset1=dataset1,
        dataset2=dataset2,
        n_jobs=n_jobs,
        verbose=verbose,
        sigma=sigma,
        compute_diagonal=True,
        be=be,
    )
    dataset1 = to_time_series_dataset(dataset1, be=be)
    if dataset2 is None:
        diagonal = be.diag(be.sqrt(1.0 / be.diag(unnormalized_matrix)))
        diagonal_left = diagonal_right = diagonal
    else:
        dataset2 = to_time_series_dataset(dataset2, be=be)
        diagonal_left = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
            delayed(unnormalized_gak)(dataset1[i], dataset1[i], sigma=sigma, be=be)
            for i in range(len(dataset1))
        )
        diagonal_right = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
            delayed(unnormalized_gak)(dataset2[j], dataset2[j], sigma=sigma, be=be)
            for j in range(len(dataset2))
        )
        diagonal_left = be.diag(1.0 / be.sqrt(be.array(diagonal_left)))
        diagonal_right = be.diag(1.0 / be.sqrt(be.array(diagonal_right)))
    return diagonal_left @ unnormalized_matrix @ diagonal_right


def sigma_gak(dataset, n_samples=100, random_state=None, be=None):
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
    be : Backend object or string or None
        Backend.

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
    be = instantiate_backend(be, dataset)

    random_state = check_random_state(random_state)
    dataset = to_time_series_dataset(dataset, be=be)
    n_ts, sz, d = be.shape(dataset)
    if not check_equal_size(dataset, be=be):
        sz = be.min([ts_size(ts) for ts in dataset])
    if n_ts * sz < n_samples:
        replace = True
    else:
        replace = False
    sample_indices = random_state.choice(n_ts * sz, size=n_samples, replace=replace)
    dists = be.pdist(
        dataset[:, :sz, :].reshape((-1, d))[sample_indices],
        metric="euclidean",
    )
    return be.median(dists) * be.sqrt(sz)


def gamma_soft_dtw(dataset, n_samples=100, random_state=None, be=None):
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
    be : Backend object or string or None
        Backend.

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
    be = instantiate_backend(be, dataset)
    return (
        2.0
        * sigma_gak(
            dataset=dataset, n_samples=n_samples, random_state=random_state, be=be
        )
        ** 2
    )


def soft_dtw(ts1, ts2, gamma=1.0, be=None):
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
        Gamma parameter for Soft-DTW
    be : Backend object or string or None
        Backend.

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
    be = instantiate_backend(be, ts1)
    if gamma == 0.0:
        return dtw(ts1, ts2, be=be) ** 2
    return SoftDTW(
        SquaredEuclidean(ts1[: ts_size(ts1)], ts2[: ts_size(ts2)], be=be),
        gamma=gamma,
        be=be,
    ).compute()


def soft_dtw_alignment(ts1, ts2, gamma=1.0, be=None):
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
        Gamma parameter for Soft-DTW
    be : Backend object or string or None
        Backend.

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
    be = instantiate_backend(be, ts1)
    if gamma == 0.0:
        path, dist = dtw_path(ts1, ts2, be=be)
        dist_sq = dist**2
        a = be.zeros((ts_size(ts1), ts_size(ts2)))
        for i, j in path:
            a[i, j] = 1.0
    else:
        sdtw = SoftDTW(
            SquaredEuclidean(ts1[: ts_size(ts1)], ts2[: ts_size(ts2)], be=be),
            gamma=gamma,
            be=be,
        )
        dist_sq = sdtw.compute()
        a = sdtw.grad()
    return a, dist_sq


def cdist_soft_dtw(dataset1, dataset2=None, gamma=1.0, be=None):
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
        Gamma parameter for Soft-DTW
    be : Backend object or string or None
        Backend.

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
    be = instantiate_backend(be, dataset1)
    dataset1 = to_time_series_dataset(dataset1, dtype=be.float64, be=be)

    self_similarity = False
    if dataset2 is None:
        dataset2 = dataset1
        self_similarity = True
    else:
        dataset2 = to_time_series_dataset(dataset2, dtype=be.float64, be=be)

    dists = be.empty((dataset1.shape[0], dataset2.shape[0]))

    equal_size_ds1 = check_equal_size(dataset1, be=be)
    equal_size_ds2 = check_equal_size(dataset2, be=be)

    for i, ts1 in enumerate(dataset1):
        if equal_size_ds1:
            ts1_short = ts1
        else:
            ts1_short = ts1[: ts_size(ts1)]
        for j, ts2 in enumerate(dataset2):
            if equal_size_ds2:
                ts2_short = ts2
            else:
                ts2_short = ts2[: ts_size(ts2)]
            if self_similarity and j < i:
                dists[i, j] = dists[j, i]
            else:
                dists[i, j] = soft_dtw(ts1_short, ts2_short, gamma=gamma, be=be)

    return dists


def cdist_soft_dtw_normalized(dataset1, dataset2=None, gamma=1.0, be=None):
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
        Gamma parameter for Soft-DTW
    be : Backend object or string or None
        Backend.

    Returns
    -------
    numpy.ndarray
        Cross-similarity matrix

    Examples
    --------
    >>> time_series = np.random.randn(10, 15, 1)
    >>> np.alltrue(cdist_soft_dtw_normalized(time_series) >= 0.)
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
    be = instantiate_backend(be, dataset1)
    dists = cdist_soft_dtw(dataset1, dataset2=dataset2, gamma=gamma, be=be)
    d_ii = be.diag(dists)
    dists -= 0.5 * (be.reshape(d_ii, (-1, 1)) + be.reshape(d_ii, (1, -1)))
    return dists


class SoftDTW:
    def __init__(self, D, gamma=1.0, be=None, compute_with_backend=False):
        """
        Parameters
        ----------
        D : array-like, shape=[m, n], dtype=float64 or class computing distances with a method 'compute'
            Distances. An example of class computing distance is 'SquaredEuclidean'.
        gamma: float
            Regularization parameter.
            Lower is less smoothed (closer to true DTW).
        be : Backend object or string or None
            Backend.
        compute_with_backend : bool, default=False
            This parameter has no influence when the NumPy backend is used.
            When using a backend different from NumPy is used:
            If `True`, the computation is done with the corresponding backend.
            If `False`, a conversion to the NumPy backend can be used to accelerate the computation.

        Attributes
        ----------
        self.R_: array, shape = [m + 2, n + 2]
            Accumulated cost matrix (stored after calling `compute`).
        """
        be = instantiate_backend(be, D)
        self.be = be
        self.compute_with_backend = compute_with_backend
        if hasattr(D, "compute"):
            self.D = D.compute()
        else:
            self.D = D
        self.D = self.be.cast(self.D, dtype=self.be.float64)

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the backward recursion.
        m, n = self.be.shape(self.D)
        self.R_ = self.be.zeros((m + 2, n + 2), dtype=self.be.float64)
        self.computed = False

        self.gamma = self.be.array(gamma, dtype=self.be.float64)

    def compute(self):
        """Compute soft-DTW by dynamic programming.

        Returns
        -------
        sdtw: float
            soft-DTW discrepancy.
        """
        m, n = self.be.shape(self.D)

        if self.be.is_numpy:
            _njit_soft_dtw(self.D, self.R_, gamma=self.gamma)
        elif not self.compute_with_backend:
            _njit_soft_dtw(
                self.be.to_numpy(self.D),
                self.be.to_numpy(self.R_),
                gamma=self.be.to_numpy(self.gamma),
            )
            self.R_ = self.be.array(self.R_)
        else:
            _soft_dtw(self.D, self.R_, gamma=self.gamma, be=self.be)

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

        m, n = self.be.shape(self.D)

        # Add an extra row and an extra column to D.
        # Needed to deal with edge cases in the recursion.
        D = self.be.vstack((self.D, self.be.zeros(n)))
        D = self.be.hstack((D, self.be.zeros((m + 1, 1))))

        # Allocate memory.
        # We need +2 because we use indices starting from 1
        # and to deal with edge cases in the recursion.
        E = self.be.zeros((m + 2, n + 2), dtype=self.be.float64)

        if self.be.is_numpy:
            _njit_soft_dtw_grad(D, self.R_, E, gamma=self.gamma)
        elif not self.compute_with_backend:
            _njit_soft_dtw_grad(
                self.be.to_numpy(D),
                self.be.to_numpy(self.R_),
                self.be.to_numpy(E),
                gamma=self.be.to_numpy(self.gamma),
            )
            self.R_ = self.be.array(self.R_)
        else:
            _soft_dtw_grad(D, self.R_, E, gamma=self.gamma, be=self.be)

        return E[1:-1, 1:-1]


class SquaredEuclidean:
    def __init__(self, X, Y, be=None, compute_with_backend=False):
        """
        Parameters
        ----------
        X: array, shape = [m, d]
            First time series.
        Y: array, shape = [n, d]
            Second time series.
        be : Backend object or string or None
            Backend.
        compute_with_backend : bool, default=False
            This parameter has no influence when the NumPy backend is used.
            When using a backend different from NumPy is used:
            If `True`, the computation is done with the corresponding backend.
            If `False`, a conversion to the NumPy backend can be used to accelerate the computation.

        Examples
        --------
        >>> SquaredEuclidean([1, 2, 2, 3], [1, 2, 3, 4]).compute()
        array([[0., 1., 4., 9.],
               [1., 0., 1., 4.],
               [1., 0., 1., 4.],
               [4., 1., 0., 1.]])
        """
        self.be = instantiate_backend(be, X)
        self.compute_with_backend = compute_with_backend
        self.X = self.be.cast(to_time_series(X), dtype=self.be.float64)
        self.Y = self.be.cast(to_time_series(Y), dtype=self.be.float64)

    def compute(self):
        """Compute distance matrix.

        Returns
        -------
        D: array, shape = [m, n]
            Distance matrix.
        """
        return self.be.pairwise_euclidean_distances(self.X, self.Y) ** 2

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
        G = self.be.zeros_like(self.X, dtype=self.be.float64)

        if self.be.is_numpy:
            _njit_jacobian_product_sq_euc(self.X, self.Y, E.astype(np.float64), G)
        elif not self.compute_with_backend:
            _njit_jacobian_product_sq_euc(
                self.be.to_numpy(self.X),
                self.be.to_numpy(self.Y),
                self.be.to_numpy(E).astype(np.float64),
                self.be.to_numpy(G),
            )
            G = self.be.array(G)
        else:
            _jacobian_product_sq_euc(
                self.X, self.Y, self.be.cast(E, self.be.float64), G
            )

        return G
