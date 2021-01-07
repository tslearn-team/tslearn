from sklearn.cross_decomposition import CCA
from .dtw_variants import dtw_path
from .utils import _cdist_generic
from ..utils import to_time_series
import numpy as np

def _get_warp_matrices(warp_path):
    """
    Convert warping path sequence to matrices.

    Parameters
    ----------
    warp_path = first output of the dtw_path function = list of tuple of two 
    indices that are matched together.

    Returns
    -------
    two 2D matrices of size m x T (m = number of step to match the two 
    sequences

    """

    # number of indices for the alignment
    m = len(warp_path)
    # number of frame of each sequence
    # (for DTW, the last two indices are matched)
    nx = warp_path[-1][0] + 1
    ny = warp_path[-1][1] + 1
    Wx = np.zeros((m, nx))
    Wy = np.zeros((m, ny))

    for i, match in enumerate(warp_path):
        Wx[i, match[0]] = 1
        Wy[i, match[1]] = 1

    return Wx, Wy


def ctw_path(s1, s2, max_iter=100, n_components=None,
             global_constraint=None, sakoe_chiba_radius=None,
             itakura_max_slope=None, verbose=False):
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
    s1
        A time series.

    s2
        Another time series.
        
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
    s1 = to_time_series(s1, remove_nans=True)
    s2 = to_time_series(s2, remove_nans=True)

    if n_components is None:
        n_components = min(s1.shape[-1], s2.shape[-1])

    cca = CCA(n_components=n_components)

    # first iteration :
    # identity matrices, this relates to apply first a dtw on the
    # (possibly truncated to a fixed number of features) inputs
    seq1_tr = s1.dot(np.eye(s1.shape[1], n_components))
    seq2_tr = s2.dot(np.eye(s2.shape[1], n_components))
    current_path, score_match = dtw_path(seq1_tr, seq2_tr,
                                         global_constraint=global_constraint,
                                         sakoe_chiba_radius=sakoe_chiba_radius,
                                         itakura_max_slope=itakura_max_slope)
    current_score = score_match

    if verbose:
        print("Iteration 0, score={}".format(current_score))

    for it in range(max_iter-1):
        Wx, Wy = _get_warp_matrices(current_path)

        cca.fit(Wx.dot(s1), Wy.dot(s2))
        seq1_tr, seq2_tr = cca.transform(s1, s2)

        current_path, score_match = dtw_path(
            seq1_tr, seq2_tr,
            global_constraint=global_constraint,
            sakoe_chiba_radius=sakoe_chiba_radius,
            itakura_max_slope=itakura_max_slope
        )

        if np.array_equal(current_path, current_path):
            break

        current_score = score_match

        if verbose:
            print("Iteration {}, score={}".format(it + 1, current_score))

    return current_path, cca, current_score


def ctw(s1, s2, max_iter=100, n_components=None,
        global_constraint=None, sakoe_chiba_radius=None,
        itakura_max_slope=None, verbose=False):
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
    s1
        A time series.

    s2
        Another time series.
        
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
    return ctw_path(s1=s1, s2=s2, max_iter=max_iter,
                    n_components=n_components,
                    global_constraint=global_constraint,
                    sakoe_chiba_radius=sakoe_chiba_radius,
                    itakura_max_slope=itakura_max_slope, verbose=verbose)[-1]


def cdist_ctw(dataset1, dataset2=None, max_iter=100, n_components=None,
              global_constraint=None, sakoe_chiba_radius=None,
              itakura_max_slope=None, n_jobs=None, verbose=0):
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
    dataset1 : array-like
        A dataset of time series

    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
        
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

    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix

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
    return _cdist_generic(dist_fun=ctw, dataset1=dataset1, dataset2=dataset2,
                          n_jobs=n_jobs, verbose=verbose,
                          compute_diagonal=False,
                          max_iter=max_iter, n_components=n_components,
                          global_constraint=global_constraint,
                          sakoe_chiba_radius=sakoe_chiba_radius,
                          itakura_max_slope=itakura_max_slope)
