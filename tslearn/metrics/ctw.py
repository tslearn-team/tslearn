import numpy as np
from sklearn.cross_decomposition import CCA

from tslearn.backend import instantiate_backend

from ..utils import to_time_series
from .dtw_variants import dtw_path
from .utils import _cdist_generic


def _get_warp_matrices(warp_path, be):
    """Convert warping path sequence to matrices.

    Parameters
    ----------
    warp_path : list of tuple of two indices that are matched together, length = m
        First output of the dtw_path function.
    be : Backend object or string
        Backend.

    Returns
    -------
    Wx : array-like, shape=(m, nx)
        Matrix. The number of steps to match the two sequences is equal to m.
    Wy : array-like, shape=(m, ny)
        Matrix.
    """
    be = instantiate_backend(be, warp_path[0][0])
    # number of indices for the alignment
    m = len(warp_path)
    # number of frame of each sequence
    # (for DTW, the last two indices are matched)
    nx = warp_path[-1][0] + 1
    ny = warp_path[-1][1] + 1
    Wx = be.zeros((m, nx), dtype=be.float64)
    Wy = be.zeros((m, ny), dtype=be.float64)

    for i, match in enumerate(warp_path):
        Wx[i, match[0]] = 1
        Wy[i, match[1]] = 1

    return Wx, Wy


def ctw_path(
    s1,
    s2,
    max_iter=100,
    n_components=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    verbose=False,
    be=None,
):
    """Compute Canonical Time Warping (CTW) similarity measure between
    (possibly multidimensional) time series and return the alignment path, the
    canonical correlation analysis (sklearn) object and the similarity.

    Canonical Time Warping is a method to align time series under rigid
    registration of the feature space.
    It should not be confused with Dynamic Time Warping (DTW), though CTW uses
    DTW.

    It is not required that both time series share the same size, nor the same
    dimension (CTW will find a subspace that best aligns feature spaces).
    CTW was originally presented in [1]_.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    max_iter : int (default: 100)
        Number of iterations for the CTW algorithm. Each iteration
    n_components : int (default: None)
        Number of components to be used for Canonical Correlation Analysis.
        If None, the lower minimum number of features between s1 and s2 is
        used.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW calls.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
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
    verbose : bool (default: True)
        If True, scores are printed at each iteration of the algorithm.
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
        first index corresponds to s1 and the second one corresponds to s2
    sklearn.decomposition.CCA
        The Canonical Correlation Analysis object used to align time series
        at convergence.
    float
        Similarity score

    Examples
    --------
    >>> path, cca, dist = ctw_path([1, 2, 3], [1., 2., 2., 3.])
    >>> path
    [(0, 0), (1, 1), (1, 2), (2, 3)]
    >>> type(cca)  # doctest: +ELLIPSIS
    <class 'sklearn.cross_decomposition...CCA'>
    >>> dist
    0.0
    >>> path, cca, dist = ctw_path([1, 2, 3],
    ...                            [[1., 1.], [2., 2.], [2., 2.], [3., 3.]])
    >>> dist
    0.0

    See Also
    --------
    ctw : Get only the similarity score for CTW

    References
    ----------
    .. [1] F. Zhou and F. Torre, "Canonical time warping for alignment of
       human behavior". NIPS 2009.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = to_time_series(s1, remove_nans=True, be=be)
    s2 = to_time_series(s2, remove_nans=True, be=be)
    s1, s2 = be.array(s1, dtype=be.float64), be.array(s2, dtype=be.float64)

    if n_components is None:
        n_components = min(s1.shape[-1], s2.shape[-1])

    cca = CCA(n_components=n_components)

    # first iteration :
    # identity matrices, this relates to apply first a dtw on the
    # (possibly truncated to a fixed number of features) inputs
    seq1_tr = s1 @ be.eye(s1.shape[1], n_components, dtype=be.float64)
    seq2_tr = s2 @ be.eye(s2.shape[1], n_components, dtype=be.float64)
    current_path, score_match = dtw_path(
        seq1_tr,
        seq2_tr,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        be=be,
    )
    current_score = score_match

    if verbose:
        print("Iteration 0, score={}".format(current_score))

    for it in range(max_iter - 1):
        Wx, Wy = _get_warp_matrices(current_path, be=be)

        cca.fit(Wx @ s1, Wy @ s2)
        seq1_tr, seq2_tr = cca.transform(s1, s2)

        current_path, score_match = dtw_path(
            seq1_tr,
            seq2_tr,
            global_constraint=global_constraint,
            sakoe_chiba_radius=sakoe_chiba_radius,
            itakura_max_slope=itakura_max_slope,
            be=be,
        )

        if np.array_equal(current_path, current_path):
            break

        current_score = score_match

        if verbose:
            print("Iteration {}, score={}".format(it + 1, current_score))

    return current_path, cca, current_score


def ctw(
    s1,
    s2,
    max_iter=100,
    n_components=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    verbose=False,
    be=None,
):
    """Compute Canonical Time Warping (CTW) similarity measure between
    (possibly multidimensional) time series and return the similarity.

    Canonical Time Warping is a method to align time series under rigid
    registration of the feature space.
    It should not be confused with Dynamic Time Warping (DTW), though CTW uses
    DTW.

    It is not required that both time series share the same size, nor the same
    dimension (CTW will find a subspace that best aligns feature spaces).
    CTW was originally presented in [1]_.

    Parameters
    ----------
    s1 : array-like, shape=(sz1, d) or (sz1,)
        A time series. If shape is (sz1,), the time series is assumed to be univariate.
    s2 : array-like, shape=(sz2, d) or (sz2,)
        Another time series. If shape is (sz2,), the time series is assumed to be univariate.
    max_iter : int (default: 100)
        Number of iterations for the CTW algorithm. Each iteration
    n_components : int (default: None)
        Number of components to be used for Canonical Correlation Analysis.
        If None, the lower minimum number of features between seq1 and seq2 is
        used.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW calls.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
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
    verbose : bool (default: True)
        If True, scores are printed at each iteration of the algorithm.
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
    >>> ctw([1, 2, 3], [1., 2., 2., 3.])
    0.0
    >>> ctw([1, 2, 3], [[1., 1.], [2., 2.], [2., 2.], [3., 3.]])
    0.0

    See Also
    --------
    ctw : Get only the similarity score for CTW

    References
    ----------
    .. [1] F. Zhou and F. Torre, "Canonical time warping for alignment of
       human behavior". NIPS 2009.
    """
    be = instantiate_backend(be, s1, s2)
    s1 = be.array(s1)
    s2 = be.array(s2)
    return ctw_path(
        s1=s1,
        s2=s2,
        max_iter=max_iter,
        n_components=n_components,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        verbose=verbose,
        be=be,
    )[-1]


def cdist_ctw(
    dataset1,
    dataset2=None,
    max_iter=100,
    n_components=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
    be=None,
):
    r"""Compute cross-similarity matrix using Canonical Time Warping (CTW)
    similarity measure.

    Canonical Time Warping is a method to align time series under rigid
    registration of the feature space.
    It should not be confused with Dynamic Time Warping (DTW), though CTW uses
    DTW.

    It is not required that both time series share the same size, nor the same
    dimension (CTW will find a subspace that best aligns feature spaces).
    CTW was originally presented in [1]_.

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
    max_iter : int (default: 100)
        Number of iterations for the CTW algorithm. Each iteration
    n_components : int (default: None)
        Number of components to be used for Canonical Correlation Analysis.
        If None, the lower minimum number of features between seq1 and seq2 is
        used.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW calls.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
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
    >>> cdist_ctw([[1, 2, 2, 3], [1., 2., 3., 4.]])
    array([[0., 1.],
           [1., 0.]])
    >>> cdist_ctw([[1, 2, 2, 3], [1., 2., 3., 4.]],
    ...           [[[1, 1], [2, 2], [3, 3]], [[2, 2], [3, 3], [4, 4], [5, 5]]])
    array([[0.        , 2.44948974],
           [1.        , 1.41421356]])

    See Also
    --------
    ctw : Get CTW similarity score

    References
    ----------
    .. [1] F. Zhou and F. Torre, "Canonical time warping for alignment of
       human behavior". NIPS 2009.
    """  # noqa: E501
    be = instantiate_backend(be, dataset1, dataset2)
    return _cdist_generic(
        dist_fun=ctw,
        dataset1=dataset1,
        dataset2=dataset2,
        n_jobs=n_jobs,
        verbose=verbose,
        compute_diagonal=False,
        max_iter=max_iter,
        n_components=n_components,
        global_constraint=global_constraint,
        sakoe_chiba_radius=sakoe_chiba_radius,
        itakura_max_slope=itakura_max_slope,
        be=be,
    )
