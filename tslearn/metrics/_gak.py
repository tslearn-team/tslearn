"""GAK metric toolbox."""
import math

from numba import njit

import numpy

from joblib import Parallel, delayed

from sklearn.utils import check_random_state

from tslearn.backend import instantiate_backend
from tslearn.backend.pytorch_backend import HAS_TORCH
from tslearn.utils import to_time_series, to_time_series_dataset
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
                    \exp \left( - \frac{ \left\| x_{\pi_1(i)} - y_{\pi_2(i)} \right\|^2}{2 \sigma^2} \right)

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

    # Normalizing in log space keeps the ratio representable even when the
    # unnormalized values themselves overflow (see issue #450).
    log_denom = 0.5 * (
        _log_unnormalized_gak(s1, s1, sigma=sigma, backend=be)
        + _log_unnormalized_gak(s2, s2, sigma=sigma, backend=be)
    )

    return be.exp(
        _log_unnormalized_gak(s1, s2, sigma=sigma, backend=be) - log_denom
    )


def unnormalized_gak(s1, s2, sigma=1.0, be=None):
    r"""Compute Global Alignment Kernel (GAK) between (possibly
    multidimensional) time series and return it.

    .. math::

        k(\mathbf{x}, \mathbf{y}) =
            \sum_{\pi \in \mathcal{A}(\mathbf{x}, \mathbf{y})}
                \prod_{i=1}^{ | \pi | }
                    \exp \left( - \frac{ \left\| x_{\pi_1(i)} - y_{\pi_2(i)} \right\|^2}{2 \sigma^2} \right)

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

    Notes
    -----
    The unnormalized kernel sums over every possible alignment between the two
    series, so it grows extremely fast with their length and leaves the range
    of a 64-bit float for time series longer than about 405 samples, in which
    case `inf` is returned. Use :func:`gak`, whose normalization is computed in
    log space, when a value is needed for long time series.

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


def _log_gram_matrix(s1, s2, sigma, backend):
    if backend.is_numpy:
        log_gram = backend.cdist(s1, s2, "sqeuclidean")
        numpy.negative(log_gram, out=log_gram)
        log_gram /= 2 * sigma * sigma
        correction = numpy.exp(log_gram)
        numpy.subtract(2, correction, out=correction)
        numpy.log(correction, out=correction)
        log_gram -= correction
        return log_gram
    log_gram = -backend.cdist(s1, s2, "sqeuclidean") / (2 * sigma * sigma)
    log_gram -= backend.log(2 - backend.exp(log_gram))
    return log_gram


def _unnormalized_gak(s1, s2, sigma, backend):
    gram = backend.exp(_log_gram_matrix(s1, s2, sigma, backend))
    if backend.is_numpy:
        return _njit_gak_from_gram_matrix(gram)
    else:
        return _gak_from_gram_matrix(gram)


def _log_unnormalized_gak(s1, s2, sigma, backend):
    """Compute the natural logarithm of the unnormalized GAK value.

    The unnormalized kernel sums over every alignment path, so it grows like
    the central Delannoy numbers (:math:`\\approx (3 + 2\\sqrt{2})^{sz}`) and
    leaves the range of a 64-bit float for time series longer than about 405
    samples. Use linear space first unless a lower bound proves overflow,
    and fall back to log space whenever the linear result is not usable.
    """
    log_gram = _log_gram_matrix(s1, s2, sigma, backend)
    if backend.is_numpy:
        gak_from_gram = _njit_gak_from_gram_matrix
        log_gak_from_gram = _njit_log_gak_from_gram_matrix
        n, m = log_gram.shape
        log_max = math.log(numpy.finfo(log_gram.dtype).max)
        # The Delannoy upper bound rules out overflow for short series.
        if n and m and (n + m - 2) * math.log(1 + math.sqrt(2)) > log_max:
            # Count just paths with k diagonal steps. Every path has at most
            # n + m - 1 cells, each weighted by at least exp(min(log_gram)).
            # This lower bound proves overflow without a linear recurrence.
            k = min(n - 1, m - 1) // 2
            log_paths = (
                math.lgamma(n + m - 1 - k)
                - math.lgamma(n - k)
                - math.lgamma(m - k)
                - math.lgamma(k + 1)
            )
            if log_paths + (n + m - 1) * log_gram.min() > log_max:
                return log_gak_from_gram(log_gram)
    else:
        gak_from_gram = _gak_from_gram_matrix
        log_gak_from_gram = _log_gak_from_gram_matrix

    value = gak_from_gram(backend.exp(log_gram))
    if 0.0 < value < backend.inf:
        return backend.log(value)

    # The value has overflowed to `inf` (or underflowed to 0): redo the
    # recursion in log space, where it stays representable.
    return log_gak_from_gram(log_gram)


def __make_gak_from_gram_matrix(backend):

    if backend is numpy:
        @njit(nogil=True)
        def _gak_from_gram_matrix_rolling(gram):
            sz1, sz2 = gram.shape
            row = numpy.zeros(sz2 + 1, dtype=gram.dtype)
            row[0] = 1.0
            for i in range(sz1):
                diag = row[0]
                row[0] = 0.0
                for j in range(sz2):
                    top = row[j + 1]
                    row[j + 1] = (top + row[j] + diag) * gram[i, j]
                    diag = top
            return row[-1]

        return _gak_from_gram_matrix_rolling

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


def __make_log_gak_from_gram_matrix(backend):

    if backend is numpy:
        @njit(nogil=True)
        def _log_gak_from_gram_matrix_rolling(log_gram, gram):
            sz1, sz2 = log_gram.shape
            neg_inf = -numpy.inf
            if gram.size:
                # Scale cell (i, j) by r ** (i + j), where 2*r + r*r = 1.
                # For weights <= 1 the recurrence is then a convex sum:
                # intermediates stay bounded without per-cell logarithms.
                scale = numpy.sqrt(2.0) - 1.0
                scale_squared = scale * scale
                row = numpy.zeros(sz2 + 1, dtype=log_gram.dtype)
                row[0] = 1.0
                safe = True
                floor = numpy.finfo(log_gram.dtype).tiny
                # The unweighted scaled recurrence is at least
                # scale ** (max(i, j) + 1) for one-based cell indices.
                # Every path uses at most sz1 + sz2 - 1 weights, so this
                # bound proves that every intermediate stays normal.
                lower_bound = (
                    (sz1 + sz2 - 1) * numpy.min(log_gram)
                    + (max(sz1, sz2) + 1) * numpy.log(scale)
                )
                if lower_bound > numpy.log(floor) + numpy.log(2.0):
                    gram *= scale
                    for i in range(sz1):
                        diag = row[0]
                        row[0] = 0.0
                        left = 0.0
                        for j in range(sz2):
                            top = row[j + 1]
                            value = (top + left + diag * scale) * gram[i, j]
                            row[j + 1] = value
                            left = value
                            diag = top
                    return numpy.log(row[-1]) - (sz1 + sz2) * numpy.log(scale)
                for i in range(sz1):
                    diag = row[0]
                    row[0] = 0.0
                    left = 0.0
                    for j in range(sz2):
                        top = row[j + 1]
                        value = (
                            (top + left) * scale + diag * scale_squared
                        ) * gram[i, j]
                        # Discard this attempt if scaling enters the
                        # subnormal range, where relative accuracy degrades.
                        if value < floor:
                            safe = False
                            break
                        row[j + 1] = value
                        left = value
                        diag = top
                    if not safe:
                        break
                if safe:
                    return numpy.log(row[-1]) - (sz1 + sz2) * numpy.log(scale)

            row = numpy.full(sz2 + 1, neg_inf, dtype=log_gram.dtype)
            row[0] = 0.0
            for i in range(sz1):
                diag = row[0]
                row[0] = neg_inf
                for j in range(sz2):
                    top = row[j + 1]
                    left = row[j]
                    # One predecessor contributes exp(0) = 1 exactly.
                    max_pred = top
                    other = left
                    if left > max_pred:
                        max_pred = left
                        other = top
                    if diag > max_pred:
                        max_pred, diag = diag, max_pred
                    if max_pred == neg_inf:
                        row[j + 1] = neg_inf
                    else:
                        row[j + 1] = (
                            max_pred
                            + numpy.log(
                                1.0 + numpy.exp(other - max_pred)
                                + numpy.exp(diag - max_pred)
                            )
                            + log_gram[i, j]
                        )
                    diag = top
            return row[-1]

        def _log_gak_from_gram_matrix_numpy(log_gram):
            # NumPy evaluates the independent exponentials in vectorized loops.
            # Keep the dependent recurrence compiled, using its private buffer.
            if log_gram.size and numpy.max(log_gram) <= 0.0:
                gram = numpy.exp(log_gram)
            else:
                gram = numpy.empty((0, 0), dtype=log_gram.dtype)
            return _log_gak_from_gram_matrix_rolling(log_gram, gram)

        return _log_gak_from_gram_matrix_numpy

    def _log_gak_from_gram_matrix_generic(
        log_gram
    ):

        sz1, sz2 = log_gram.shape
        neg_inf = -backend.inf

        cum_sum = backend.full(
            (sz1 + 1, sz2 + 1), neg_inf, dtype=log_gram.dtype
        )
        cum_sum[0, 0] = 0.0

        for i in range(sz1):
            for j in range(sz2):
                # log-sum-exp of the three predecessors, factoring out their
                # maximum so that the exponentials stay in [0, 1].
                top = cum_sum[i, j + 1]
                left = cum_sum[i + 1, j]
                diag = cum_sum[i, j]

                max_pred = top
                if left > max_pred:
                    max_pred = left
                if diag > max_pred:
                    max_pred = diag

                if max_pred == neg_inf:
                    # Every path reaching this cell has zero weight.
                    continue

                cum_sum[i + 1, j + 1] = (
                    max_pred
                    + backend.log(
                        backend.exp(top - max_pred)
                        + backend.exp(left - max_pred)
                        + backend.exp(diag - max_pred)
                    )
                    + log_gram[i, j]
                )

        return cum_sum[-1, -1]

    if backend is numpy:
        return njit(nogil=True)(_log_gak_from_gram_matrix_generic)
    else:
        return _log_gak_from_gram_matrix_generic


_njit_log_gak_from_gram_matrix = __make_log_gak_from_gram_matrix(numpy)
if HAS_TORCH:
    _log_gak_from_gram_matrix = __make_log_gak_from_gram_matrix(
        instantiate_backend("torch")
    )
else:
    _log_gak_from_gram_matrix = _njit_log_gak_from_gram_matrix


def cdist_gak(
    dataset1,
    dataset2=None,
    sigma=1.0,
    n_jobs=None,
    verbose=0,
    be=None
):
    r"""Compute cross-similarity matrix using Global Alignment kernel (GAK).
    Note that GAK is a kernel, the larger GAK values mean more similar time series.

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
        ``-1`` means using all processors. See scikit-learn's
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

    log_unnormalized_matrix = _cdist_generic(
       dist_fun=_log_unnormalized_gak,
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
       log_diagonal_left = be.diag(log_unnormalized_matrix)
       log_diagonal_right = log_diagonal_left
    else:
       log_diagonal_left = Parallel(
           n_jobs=n_jobs, prefer="threads", verbose=verbose
       )(
           delayed(_log_unnormalized_gak)(
               _to_time_series(dataset1[i], remove_nans=True, backend=be),
               _to_time_series(dataset1[i], remove_nans=True, backend=be),
               sigma=sigma,
               backend=be
           )
           for i in range(len(dataset1))
       )
       log_diagonal_right = Parallel(
           n_jobs=n_jobs, prefer="threads", verbose=verbose
       )(
           delayed(_log_unnormalized_gak)(
               _to_time_series(dataset2[j], remove_nans=True, backend=be),
               _to_time_series(dataset2[j], remove_nans=True, backend=be),
               sigma=sigma,
               backend=be
           )
           for j in range(len(dataset2))
       )
       log_diagonal_left = be.array(log_diagonal_left)
       log_diagonal_right = be.array(log_diagonal_right)

    # Normalize in log space so that the result stays finite even when the
    # unnormalized kernel values overflow (see issue #450).
    return be.exp(
       log_unnormalized_matrix
       - 0.5 * log_diagonal_left[:, None]
       - 0.5 * log_diagonal_right[None, :]
    )
