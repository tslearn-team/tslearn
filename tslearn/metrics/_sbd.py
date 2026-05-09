"""Public Shape-Based Distance (SBD) helpers.

SBD is the distance used inside :class:`tslearn.clustering.KShape`. Until now
it was only available indirectly via :func:`cdist_normalized_cc`. Issue #276
asks for a function-level handle, the way :func:`tslearn.metrics.dtw` exposes
DTW as a stand-alone distance.

The implementation defers to the existing numba-jitted
:func:`tslearn.metrics.cycc.normalized_cc` so behaviour matches what KShape
already does internally — see ``KShape._cross_dists`` in
``tslearn/clustering/kshape.py``.
"""
import numpy

from .cycc import cdist_normalized_cc, normalized_cc
from ..utils import to_time_series, to_time_series_dataset


def sbd(s1, s2):
    r"""Shape-Based Distance (SBD) between two time series.

    SBD is defined in [1]_ as

    .. math::

        \mathrm{SBD}(\mathbf{x}, \mathbf{y}) =
            1 - \max_{w}\;\frac{\mathrm{NCC}_w(\mathbf{x}, \mathbf{y})}
                                {\|\mathbf{x}\|_2 \cdot \|\mathbf{y}\|_2}

    where :math:`\mathrm{NCC}_w` denotes the cross-correlation of the two
    series at lag :math:`w`. SBD is the distance used by
    :class:`tslearn.clustering.KShape`.

    Parameters
    ----------
    s1 : array-like, shape=(sz, d) or (sz,)
        A time series.
    s2 : array-like, shape=(sz, d) or (sz,)
        Another time series of the same length and dimensionality as ``s1``.

    Returns
    -------
    float
        SBD value in :math:`[0, 2]`. ``0`` means perfect shape match (up to a
        cyclic shift), ``2`` means perfectly anti-correlated.

    Examples
    --------
    >>> import numpy
    >>> float(sbd([1., 2., 3.], [1., 2., 3.]))
    0.0
    >>> # Equal-length series with a partial shape match
    >>> float(round(sbd([1., 2., 3.], [3., 2., 1.]), 4))
    0.1429

    See Also
    --------
    cdist_sbd : Pairwise SBD on two datasets.
    cdist_normalized_cc : Underlying cross-correlation matrix.
    tslearn.clustering.KShape : Clustering algorithm built on SBD.

    References
    ----------
    .. [1] J. Paparrizos and L. Gravano. k-Shape: Efficient and Accurate
       Clustering of Time Series. SIGMOD 2015.
    """
    # SBD requires matching length / dimensionality. Use to_time_series so
    # callers can pass plain Python lists or 1-D arrays just like dtw().
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    if s1.shape != s2.shape:
        raise ValueError(
            "sbd() requires both time series to have the same shape, "
            f"got {s1.shape} and {s2.shape}."
        )
    # normalized_cc returns the full lag-correlation vector; SBD is 1 minus
    # its max, matching KShape._cross_dists.
    cc = normalized_cc(s1.astype(numpy.float64), s2.astype(numpy.float64))
    return 1.0 - cc.max()


def cdist_sbd(dataset1, dataset2=None):
    """Pairwise Shape-Based Distance between two time-series datasets.

    Parameters
    ----------
    dataset1 : array-like, shape=(n_ts1, sz, d) or (n_ts1, sz)
        First dataset of time series.
    dataset2 : array-like, shape=(n_ts2, sz, d) or (n_ts2, sz), optional
        Second dataset. If ``None`` (default), pairwise SBD is computed within
        ``dataset1``.

    Returns
    -------
    numpy.ndarray of shape (n_ts1, n_ts2)
        Pairwise SBD values. Same convention as :func:`sbd`: ``0`` means
        identical shape, ``2`` means perfectly anti-correlated.

    Examples
    --------
    >>> import numpy
    >>> X = numpy.array([[[1.], [2.], [3.]], [[1.], [2.], [3.]]])
    >>> dists = cdist_sbd(X)
    >>> dists.shape
    (2, 2)
    >>> float(dists[0, 1])
    0.0

    See Also
    --------
    sbd : Scalar SBD between two time series.
    cdist_normalized_cc : Underlying cross-correlation matrix.
    """
    dataset1 = to_time_series_dataset(dataset1).astype(numpy.float64)
    if dataset2 is None:
        dataset2 = dataset1
    else:
        dataset2 = to_time_series_dataset(dataset2).astype(numpy.float64)
    # cdist_normalized_cc expects pre-allocated norm vectors; passing -1.0
    # tells the kernel to compute them on the fly. We always pass
    # self_similarity=False — the self_similarity=True branch of that kernel
    # is tailored for KShape (zero diagonal, lower-triangle fill) and would
    # silently produce SBD=1 on the diagonal, which is wrong for callers who
    # want a true distance matrix where diag(SBD) == 0.
    norms1 = numpy.full(dataset1.shape[0], -1.0, dtype=numpy.float64)
    norms2 = numpy.full(dataset2.shape[0], -1.0, dtype=numpy.float64)
    cc = cdist_normalized_cc(
        dataset1, dataset2, norms1, norms2, False
    )
    return 1.0 - cc
