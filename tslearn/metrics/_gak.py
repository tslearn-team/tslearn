"""GAK metric toolbox."""
import math

from numba import njit

import numpy

from joblib import Parallel, delayed

from sklearn.utils import check_random_state

from tslearn.backend import instantiate_backend
from tslearn.backend.pytorch_backend import HAS_TORCH
from tslearn.utils import  to_time_series, to_time_series_dataset
from tslearn.utils.utils import _to_time_series

from .utils import _cdist_generic


def sigma_gak(dataset, n_samples=100, random_state=None, be=None):
    r"""Compute sigma value to be used for GAK.

    This method was originally presented in [1]_.

    Parameters
    ----------
    dataset : array-like, shape=(n_ts, sz, d) or (n_ts, sz1) or (sz,)
        A dataset of time series.
        If shape is (n_ts, sz), the dataset is composed of univariate time series.
        If shape is (sz,), the dataset is composed of a unique univariate time series.
    n_samples : int (default: 100)
        Number of samples on which median distance should be estimated.
    random_state : integer or numpy.RandomState or None (default: None)
        The generator used to draw the samples. If an integer is given, it
        fixes the seed. Defaults to the global numpy random number generator.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    float
        Suggested bandwidth (:math:`\sigma`) for the Global Alignment kernel.

    Examples
    --------
    >>> dataset = [[1, 2, 2, 3], [1., 2., 3., 4.]]
    >>> float(sigma_gak(dataset=dataset,
    ...                 n_samples=200,
    ...                 random_state=0))  # doctest: +ELLIPSIS
    2.0...

    See Also
    --------
    gak : Compute Global Alignment kernel
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels,"
       ICML 2011.
    """
    be = instantiate_backend(be, dataset)
    dataset = to_time_series_dataset(dataset, be=be)

    return _sigma_gak(dataset, n_samples, random_state, be)


def _sigma_gak(dataset, n_samples=100, random_state=None, be=None):
    if be is None:
        be = instantiate_backend(dataset)

    _, sz, d = dataset.shape

    # Remove points with nans from dataset
    dataset = dataset.reshape((-1, d))
    mask = be.isfinite(be.sum(dataset, axis=-1))
    dataset = dataset[mask]

    random_state = check_random_state(random_state)
    nb_valid_samples = len(dataset)
    replace = nb_valid_samples < n_samples
    sample_indices = random_state.choice(
        nb_valid_samples,
        size=n_samples,
        replace=replace
    )
    dists = be.pdist(
        dataset[sample_indices],
        metric="euclidean",
    )
    return be.median(dists) * be.sqrt(sz)


def gak(s1, s2, sigma=1.0, be=None):
    r"""Compute Global Alignment Kernel (GAK) between (possibly
    multidimensional) time series and return it.

    .. math::

        \text{gak}(\mathbf{x}, \mathbf{y}) =
            \frac{k(\mathbf{x}, \mathbf{y})}
                {\sqrt{k(\mathbf{x}, \mathbf{x})k(\mathbf{y}, \mathbf{y})}}

    where

    .. math::

        k(\mathbf{x}, \mathbf{y}) =
            \sum_{\pi \in \mathcal{A}(\mathbf{x}, \mathbf{y})}
                \prod_{i=1}^{ | \pi | }
                    \exp \left( - \frac{ \left\| x_{\pi_1(i)} - y_{\pi_2{j}} \right\|^2}{2 \sigma^2} \right)

    It is not required that both time series share the same size, but they must
    be the same dimension. GAK was originally presented in [1]_.
    This is a normalized version that ensures that :math:`gak(x,x)=1` for all
    :math:`x` and :math:`gak(x,y) \in [0, 1]` for all :math:`x, y`.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series.
        If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series.
        If shape is (sz2,), the time series is assumed to be univariate.
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    float
        Kernel value

    Examples
    --------
    >>> float(gak([1, 2, 3], [1., 2., 2., 3.], sigma=2.))  # doctest: +ELLIPSIS
    0.839...
    >>> float(gak([1, 2, 3], [1., 2., 2., 3., 4.]))  # doctest: +ELLIPSIS
    0.273...

    See Also
    --------
    cdist_gak : Compute cross-similarity matrix using Global Alignment kernel

    References
    ----------
    .. [1] M. Cuturi, "Fast global alignment kernels,"
       ICML 2011.
    """
    if math.isclose(sigma, 0.0):
        raise ZeroDivisionError("Sigma must be non-zero.")

    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    denom = be.sqrt(
        _unnormalized_gak(s1, s1, sigma=sigma, backend=be)
    ) * be.sqrt(
        _unnormalized_gak(s2, s2, sigma=sigma, backend=be)
    )

    return _unnormalized_gak(s1, s2, sigma=sigma, backend=be) / denom


def unnormalized_gak(s1, s2, sigma=1.0, be=None):
    r"""Compute Global Alignment Kernel (GAK) between (possibly
    multidimensional) time series and return it.

    .. math::

        k(\mathbf{x}, \mathbf{y}) =
            \sum_{\pi \in \mathcal{A}(\mathbf{x}, \mathbf{y})}
                \prod_{i=1}^{ | \pi | }
                    \exp \left( - \frac{ \left\| x_{\pi_1(i)} - y_{\pi_2{j}} \right\|^2}{2 \sigma^2} \right)

    It is not required that both time series share the same size, but they must
    be the same dimension. GAK was originally presented in [1]_.
    This is an unnormalized version.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series.
        If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series.
        If shape is (sz2,), the time series is assumed to be univariate.
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

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
    .. [1] M. Cuturi, "Fast global alignment kernels,"
       ICML 2011.
    """

    if math.isclose(sigma, 0.0):
        raise ZeroDivisionError("Sigma must be non-zero.")

    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)

    return _unnormalized_gak(s1, s2, sigma, be)


def _unnormalized_gak(s1, s2, sigma, backend):
    gram = -backend.cdist(s1, s2, "sqeuclidean") / (2 * sigma * sigma)
    gram -= backend.log(2 - backend.exp(gram))
    gram = backend.exp(gram)
    if backend.is_numpy:
        return _njit_gak_from_gram_matrix(gram)
    else:
        return _gak_from_gram_matrix(gram)


def __make_gak_from_gram_matrix(backend):

    def _gak_from_gram_matrix_generic(
        gram
    ):

        sz1, sz2 = gram.shape

        cum_sum = backend.zeros((sz1 + 1, sz2 + 1), dtype=gram.dtype)
        cum_sum[0, 0] = 1.0

        for i in range(sz1):
            for j in range(sz2):
                cum_sum[i + 1, j + 1] = (
                    cum_sum[i, j + 1] + cum_sum[i + 1, j] + cum_sum[i, j]
                ) * gram[i, j]

        return cum_sum[-1, -1]

    if backend is numpy:
        return njit(nogil=True)(_gak_from_gram_matrix_generic)
    else:
        return _gak_from_gram_matrix_generic


_njit_gak_from_gram_matrix = __make_gak_from_gram_matrix(numpy)
if HAS_TORCH:
    _gak_from_gram_matrix = __make_gak_from_gram_matrix(instantiate_backend("torch"))
else:
    _gak_from_gram_matrix = _njit_gak_from_gram_matrix


def cdist_gak(
    dataset1,
    dataset2=None,
    sigma=1.0,
    n_jobs=None,
    verbose=0,
    be=None
):
    r"""Compute cross-similarity matrix using Global Alignment kernel (GAK).

    GAK was originally presented in [1]_.

    Parameters
    ----------
    dataset1 : array-like, shape=(n_ts1, sz1, d) or (n_ts1, sz1) or (sz1,)
        A dataset of time series.
        If shape is (n_ts1, sz1), the dataset is composed of univariate time series.
        If shape is (sz1,), the dataset is composed of a unique univariate time series.
    dataset2 : None or array-like, shape=(n_ts2, sz2, d) or (n_ts2, sz2) or (sz2,) (default: None)
        Another dataset of time series.
        If `None`, self-similarity of `dataset1` is returned.
        If shape is (n_ts2, sz2), the dataset is composed of univariate time series.
        If shape is (sz2,), the dataset is composed of a unique univariate time series.
    sigma : float (default 1.)
        Bandwidth of the internal gaussian kernel used for GAK
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`__
        for more details.
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.

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
    .. [1] M. Cuturi, "Fast global alignment kernels,"
       ICML 2011.
    """  # noqa: E501
    be = instantiate_backend(be, dataset1, dataset2)
    dataset1 = to_time_series_dataset(dataset1, be=be)
    if dataset2 is not None:
        dataset2 = to_time_series_dataset(dataset2, be=be)
    return _cdist_gak(
        dataset1=dataset1,
        dataset2=dataset2,
        sigma=sigma,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be
    )


def _cdist_gak(
    dataset1,
    dataset2=None,
    sigma=1.0,
    n_jobs=None,
    verbose=0,
    be=None
):

    if math.isclose(sigma, 0.0):
       raise ZeroDivisionError("Sigma must be non-zero.")

    if be is None:
       be = instantiate_backend(dataset1, dataset2)

    unnormalized_matrix = _cdist_generic(
       dist_fun=_unnormalized_gak,
       dataset1=dataset1,
       dataset2=dataset2,
       n_jobs=n_jobs,
       verbose=verbose,
       compute_diagonal=True,
       be=be,
       # Distance function arguments
       backend=be,
       sigma = sigma
    )
    if dataset2 is None:
       diagonal = be.diag(be.sqrt(1.0 / be.diag(unnormalized_matrix)))
       diagonal_left = diagonal_right = diagonal
    else:
       diagonal_left = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
           delayed(_unnormalized_gak)(
               _to_time_series(dataset1[i], remove_nans=True, backend=be),
               _to_time_series(dataset1[i], remove_nans=True, backend=be),
               sigma=sigma,
               backend=be
           )
           for i in range(len(dataset1))
       )
       diagonal_right = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
           delayed(_unnormalized_gak)(
               _to_time_series(dataset2[j], remove_nans=True, backend=be),
               _to_time_series(dataset2[j], remove_nans=True, backend=be),
               sigma=sigma,
               backend=be
           )
           for j in range(len(dataset2))
       )
       diagonal_left = be.diag(1.0 / be.sqrt(diagonal_left))
       diagonal_right = be.diag(1.0 / be.sqrt(diagonal_right))

    return diagonal_left @ unnormalized_matrix @ diagonal_right
