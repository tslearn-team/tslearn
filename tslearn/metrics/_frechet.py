"""Frechet metric toolbox."""

import numpy

from numba import njit

from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series, to_time_series_dataset
from tslearn.utils.utils import _to_time_series

from ._dtw import (
    _njit_compute_path,
    _compute_path
)
from ._masks import (
    GLOBAL_CONSTRAINT_CODE,
    _njit_compute_mask,
    _compute_mask
)
from .utils import _cdist_generic


def frechet(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None
):
    r"""Compute Frechet similarity [1]_ measure between
        (possibly multidimensional) time series and return it.

        Frechet similarity score is computed as the maximum distance between
        aligned time series, i.e., if :math:`\pi` is an optimal alignment path:

        .. math::

            Frechet(X, Y) = \max_{(i, j) \in \pi} \|X_{i} - Y_{j}\|

        Note that this formula is still valid for the multivariate case.

        It is not required that both time series share the same size, but they must
        be the same dimension.

        Parameters
        ----------
        s1 : array-like, shape=(sz1, d) or (sz1,)
            A time series. If shape is (sz1,), the time series is assumed to be univariate.

        s2 : array-like, shape=(sz2, d) or (sz2,)
            Another time series. If shape is (sz2,), the time series is assumed to be univariate.

        global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
            Global constraint to restrict admissible paths for Frechet distance.

        sakoe_chiba_radius : int or None (default: None)
            Radius to be used for Sakoe-Chiba band global constraint.
            The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [2]_,
            it controls how far in time we can go in order to match a given
            point from one time series to a point in another time series.
            If None and `global_constraint` is set to "sakoe_chiba", a radius of
            1 is used.
            If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
            `global_constraint` is used to infer which constraint to use among the
            two. In this case, if `global_constraint` corresponds to no global
            constraint, a `RuntimeWarning` is raised and no global constraint is
            used.

        itakura_max_slope : float or None (default: None)
            Maximum slope for the Itakura parallelogram constraint.
            If None and `global_constraint` is set to "itakura", a maximum slope
            of 2. is used.
            If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
            `global_constraint` is used to infer which constraint to use among the
            two. In this case, if `global_constraint` corresponds to no global
            constraint, a `RuntimeWarning` is raised and no global constraint is
            used.

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
            Similarity score

        Examples
        --------
        >>> float(frechet([1, 2, 3], [1., 2., 2., 3.]))
        0.0
        >>> float(frechet([1, 2, 3], [1., 2., 2., 3., 4.]))
        1.0

        The PyTorch backend can be used to compute gradients:

        >>> import torch
        >>> s1 = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
        >>> s2 = torch.tensor([[3.0], [4.0], [-3.0]])
        >>> sim = frechet(s1, s2, be="pytorch")
        >>> print(sim)
        tensor(6., grad_fn=<SelectBackward0>)
        >>> sim.backward()
        >>> print(s1.grad)
        tensor([[0.],
                [0.],
                [1.]])

        >>> s1_2d = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], requires_grad=True)
        >>> s2_2d = torch.tensor([[3.0, 3.0], [4.0, 4.0], [-3.0, -3.0]])
        >>> sim = frechet(s1_2d, s2_2d, be="pytorch")
        >>> print(sim)
        tensor(8.4853, grad_fn=<SelectBackward0>)
        >>> sim.backward()
        >>> print(s1_2d.grad)
        tensor([[0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.7071, 0.7071]])

        See Also
        --------
        frechet_path : Get both the matching path and the similarity score
        frechet_path_from_metric : Compute similarity score and path
            using a user-defined distance metric
        cdist_frechet : Cross similarity matrix between time series datasets

        References
        ----------
        .. [1] FRÉCHET, M. "Sur quelques points du calcul fonctionnel. 
           Rendiconti del Circolo Mathematico di Palermo", 22, 1–74, 1906.
        .. [2] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """  # noqa: E501
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)
    return _frechet(
        s1,
        s2,
        global_constraint,
        sakoe_chiba_radius,
        itakura_max_slope,
        be
    )[1]


def _frechet(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
    return_path=False,
    metric="euclidean",
    **kwds
):
    if be is None:
        be = instantiate_backend(s1, s2)

    if metric != "precomputed":

        if s1.shape[0] == 0 or s2.shape[0] == 0:
            raise ValueError(
                "One of the input time series contains only nans or has zero length."
            )

        if s1.shape[1] != s2.shape[1]:
           raise ValueError("All input time series must have the same feature size.")

    else:
        s2 = s1.T

    compute_mask_ = _njit_compute_mask if be.is_numpy else _compute_mask
    mask = compute_mask_(
        len(s1),
        len(s2),
        GLOBAL_CONSTRAINT_CODE[global_constraint],
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope
    )

    distance_matrix = None
    if metric == "precomputed":
        distance_matrix = s1
    elif metric not in ["euclidean", "sqeuclidean"] and be.is_numpy:
        distance_matrix = be.pairwise_distances(s1, s2, metric=metric, **kwds)

    if distance_matrix is not None:
        if be.is_numpy:
            acc_matrix = _njit_frechet_accumulated_matrix_from_distance_matrix(distance_matrix, mask)
        else:
            acc_matrix = _frechet_accumulated_matrix_from_distance_matrix(distance_matrix, mask, be)
    else:
        if be.is_numpy:
            acc_matrix = _njit_frechet_accumulated_matrix(s1, s2, mask, squared=metric=="sqeuclidean")
        else:
            acc_matrix = _frechet_accumulated_matrix(s1, s2, mask, be, metric, **kwds)

    path = None
    if return_path:
        if be.is_numpy:
            path = _njit_compute_path(acc_matrix)
        else:
            path = _compute_path(acc_matrix)

    return path, acc_matrix[-1, -1]


@njit
def _njit_frechet_accumulated_matrix(s1, s2, mask, squared=True):
    l1, l2 = s1.shape[0], s2.shape[0]

    acc_matrix = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    acc_matrix[0, 0] = 0

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                local_distance = numpy.linalg.norm(s1[i] - s2[j])
                if squared:
                    local_distance = local_distance ** 2
                acc_matrix[i + 1, j + 1] = max(
                    local_distance,
                    min(acc_matrix[i, j + 1],
                        acc_matrix[i + 1, j],
                        acc_matrix[i, j])
                )

    return acc_matrix[1:, 1:]


def _frechet_accumulated_matrix(s1, s2, mask, backend, metric, **kwds):
    l1, l2 = s1.shape[0], s2.shape[0]

    acc_matrix = backend.full((l1 + 1, l2 + 1), numpy.inf)
    acc_matrix[0, 0] = 0

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                local_distance = (
                    backend.pairwise_distances(s1[i].reshape(1, -1),
                                               s2[j].reshape(1, -1),
                                               metric=metric,
                                               **kwds)
                )
                acc_matrix[i + 1, j + 1] = max(
                    local_distance,
                    min(acc_matrix[i, j + 1],
                        acc_matrix[i + 1, j],
                        acc_matrix[i, j])
                )
    return acc_matrix[1:, 1:]


@njit
def _njit_frechet_accumulated_matrix_from_distance_matrix(distance_matrix, mask):
    l1, l2 = distance_matrix.shape[0], distance_matrix.shape[1]

    acc_matrix = numpy.full((l1 + 1, l2 + 1), numpy.inf)
    acc_matrix[0, 0] = 0

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                acc_matrix[i + 1, j + 1] = max(
                    distance_matrix[i, j],
                    min(acc_matrix[i, j + 1],
                        acc_matrix[i + 1, j],
                        acc_matrix[i, j])
                )

    return acc_matrix[1:, 1:]


def _frechet_accumulated_matrix_from_distance_matrix(distance_matrix, mask, backend):
    l1, l2 = distance_matrix.shape[0], distance_matrix.shape[1]

    acc_matrix = backend.full((l1 + 1, l2 + 1), numpy.inf)
    acc_matrix[0, 0] = 0

    for i in range(l1):
        for j in range(l2):
            if mask[i, j]:
                acc_matrix[i + 1, j + 1] = max(
                    distance_matrix[i, j],
                    min(acc_matrix[i, j + 1],
                        acc_matrix[i + 1, j],
                        acc_matrix[i, j])
                )
    return acc_matrix[1:, 1:]


def frechet_path(
    s1,
    s2,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
):
    r"""Compute Frechet similarity measure [1]_ between
    (possibly multidimensional) time series and an optimal alignment path.

    Frechet distance is computed as the maximium distance between aligned time series,
    i.e., if :math:`\pi` is the optimal alignment path:

    .. math::

        Frechet(X, Y) = \max_{(i, j) \in \pi} \|X_{i} - Y_{j}\|

    It is not required that both time series share the same size, but they must
    be the same dimension.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for Frechet distance.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [2]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.

    float
        Similarity score

    Examples
    --------
    >>> path, dist = frechet_path([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> float(dist)
    0.0
    >>> float(frechet_path([1, 2, 3], [1., 2., 2., 3., 4.])[1])
    1.0

    See Also
    --------
    frechet : Get only the similarity score
    frechet_path_from_metric : Compute similarity score and path
        using a user-defined distance metric
    cdist_frechet : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] FRÉCHET, M. "Sur quelques points du calcul fonctionnel.
           Rendiconti del Circolo Mathematico di Palermo", 22, 1–74, 1906.
    .. [2] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.

    """  # noqa: E501
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)
    return _frechet(
        s1,
        s2,
        global_constraint,
        sakoe_chiba_radius,
        itakura_max_slope,
        be,
        return_path=True
    )


def frechet_path_from_metric(
    s1,
    s2=None,
    metric="precomputed",
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    be=None,
    **kwds
):
    r"""Compute Frechet similarity measure and an optimal alignment path [1]_
    between (possibly multidimensional) time series using a distance metric
    defined by the user.

    It is not required that both time series share the same size, but they must
    be the same dimension.

    When using Pytorch backend only "precomputed", "euclidean", "sqeuclidean"
    and callable metrics are available.
    Otherwise, valid values for metric are the same as for scikit-learn
    `pairwise_distances`_ function i.e. a string (e.g. "euclidean",
    "sqeuclidean", "hamming") or a function that is used to compute the
    pairwise distances. See `scikit`_ and `scipy`_ documentations for more
    information about the available metrics.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,) if metric!="precomputed", (sz1, sz2) otherwise
        A time series or an array of pairwise distances between samples.
        If shape is (sz1,), the time series is assumed to be univariate.

    s2 : array-like, shape=(sz2, d) or (sz2,), optional (default: None)
        A second time series, only used if metric != "precomputed".
        If shape is (sz2,), the time series is assumed to be univariate.

    metric : string or callable (default: "precomputed")
        If metric is "precomputed", `s1` is assumed to be a distance matrix.

        Otherwise, function used to compute the pairwise distances between each
        points of `s1` and `s2`.
        If metric is a string, it must be one of the options compatible
        with sklearn.metrics.pairwise_distances.
        Alternatively, if metric is a callable function, it is called on pairs
        of rows of `s1` and `s2`. The callable should take two 1 dimensional
        arrays as input and return a value indicating the distance between
        them.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for Frechet.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [2]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    **kwds
        Additional arguments to pass to sklearn pairwise_distances to compute
        the pairwise distances.

    Returns
    -------
    list of integer pairs
        Matching path represented as a list of index pairs. In each pair, the
        first index corresponds to s1 and the second one corresponds to s2.

    float
        Similarity score (sum of metric along the wrapped time series).

    Examples
    --------
    Lets create 2 numpy arrays to wrap:

    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> s1, s2 = rng.rand(5, 2), rng.rand(6, 2)

    The wrapping can be done by passing a string indicating the metric to pass
    to scikit-learn pairwise_distances:

    >>> x, y = frechet_path_from_metric(s1, s2,
    ...                                 metric="sqeuclidean")  # doctest: +ELLIPSIS
    >>> x, float(y)
    ([(0, 0), (1, 0), (2, 1), (2, 2), (2, 3), (3, 4), (4, 5)], 0.4365...)

    Or by defining a custom distance function:

    >>> sqeuclidean = lambda x, y: np.sum((x-y)**2)
    >>> x, y = frechet_path_from_metric(s1, s2, metric=sqeuclidean)  # doctest: +ELLIPSIS
    >>> x, float(y)
    ([(0, 0), (1, 0), (2, 1), (2, 2), (2, 3), (3, 4), (4, 5)], 0.4365...)

    Or by using a precomputed distance matrix as input:

    >>> from sklearn.metrics.pairwise import pairwise_distances
    >>> dist_matrix = pairwise_distances(s1, s2, metric="sqeuclidean")
    >>> x, y = frechet_path_from_metric(dist_matrix,
    ...                                 metric="precomputed")  # doctest: +ELLIPSIS
    >>> x, float(y)
    ([(0, 0), (1, 0), (2, 1), (2, 2), (2, 3), (3, 4), (4, 5)], 0.4365...)

    Notes
    -----
    By using a squared euclidean distance metric as shown above, the output
    path is the same as the one obtained by using frechet_path but the similarity
    score is the sum of squared distances instead of the euclidean distance.

    See Also
    --------
    frechet : Get only the similarity score
    frechet_path : Get both the matching path and the similarity score
    cdist_frechet : Cross similarity matrix between time series datasets

    References
    ----------
    .. [1] FRÉCHET, M. "Sur quelques points du calcul fonctionnel. 
       Rendiconti del Circolo Mathematico di Palermo", 22, 1–74, 1906.
    .. [2] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
       spoken word recognition," IEEE Transactions on Acoustics, Speech and
       Signal Processing, vol. 26(1), pp. 43--49, 1978.

    .. _pairwise_distances: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    .. _scikit: https://scikit-learn.org/stable/modules/metrics.html

    .. _scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    """  # noqa: E501
    be = instantiate_backend(be, s1, s2)
    if metric != "precomputed":
        s1 = to_time_series(s1, remove_nans=True, be=be)
        s2 = to_time_series(s2, remove_nans=True, be=be)
    else:
        s1 = be.array(s1)
    return _frechet(
        s1,
        s2,
        global_constraint,
        sakoe_chiba_radius,
        itakura_max_slope,
        be,
        return_path=True,
        metric=metric,
        **kwds
    )


def cdist_frechet(
    dataset1,
    dataset2=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    r"""Compute cross-similarity matrix using Frechet
    similarity measure [1]_.

    Frechet is computed as the maximum distance between aligned time series,
    i.e., if :math:`\pi` is an optimal alignment path:

    .. math::

        Frechet(X, Y) = \max_{(i, j) \in \pi} \|X_{i} - Y_{j}\|

    Note that this formula is still valid for the multivariate case.

    It is not required that time series share the same size, but they
    must be the same dimension.

    Parameters
    ----------
    dataset1 : array-like, shape=(n_ts1, sz1, d) or (n_ts1, sz1) or (sz1,)
        A dataset of time series.
        If shape is (n_ts1, sz1), the dataset is composed of univariate time series.
        If shape is (sz1,), the dataset is composed of a unique univariate time series.

    dataset2 : None or array-like, shape=(n_ts2, sz2, d) or (n_ts2, sz2) or (sz2,) (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
        If shape is (n_ts2, sz2), the dataset is composed of univariate time series.
        If shape is (sz2,), the dataset is composed of a unique univariate time series.

    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for Frechet.

    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [2]_,
        it controls how far in time we can go in order to match a given
        point from one time series to a point in another time series.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.

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
    cdist : array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.

    Examples
    --------
    >>> cdist_frechet([[1, 2, 2, 3], [1., 2., 3., 4.]])
    array([[0., 1.],
           [1., 0.]])
    >>> cdist_frechet([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 3], [2, 3, 4, 5]])
    array([[0., 2.],
           [1., 1.]])

    See Also
    --------
    frechet : Get only the similarity score
    frechet_path : Get both the matching path and the similarity score
    frechet_path_from_metric : Compute Frechet similarity score and path
        using a user-defined distance metric

    References
    ----------
    .. [1] FRÉCHET, M. "Sur quelques points du calcul fonctionnel. 
       Rendiconti del Circolo Mathematico di Palermo", 22, 1–74, 1906.
    .. [2] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """  # noqa: E501
    be = instantiate_backend(be, dataset1, dataset2)
    dataset1 = to_time_series_dataset(dataset1, be=be)
    if dataset2 is not None:
        dataset2 = to_time_series_dataset(dataset2, be=be)
    return _cdist_frechet(
        dataset1=dataset1,
        dataset2=dataset2,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        n_jobs=n_jobs,
        verbose=verbose,
        be=be,
    )


def _cdist_frechet(
    dataset1,
    dataset2=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    if be is None:
        be = instantiate_backend(dataset1, dataset2)
    # TODO: dev and use fully jitted _frechet for numpy backend
    return _cdist_generic(
        dist_fun=lambda *args, **kwargs: _frechet(*args, **kwargs)[1],
        dataset1=dataset1,
        dataset2=dataset2,
        n_jobs=n_jobs,
        verbose=verbose,
        compute_diagonal=False,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        be=be,
    )
