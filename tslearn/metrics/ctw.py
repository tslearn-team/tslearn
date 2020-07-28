from sklearn.cross_decomposition import CCA
from .dtw_variants import dtw_path
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


def ctw_path(seq1, seq2,
             max_iter=100, n_components=1,
             global_constraint=None, sakoe_chiba_radius=None,
             itakura_max_slope=None, verbose=0):
    """

    Parameters
    ----------
    seq = 2D array of size T x D. Both dimensions can be of variable size.
    niter = number of iteration (no early stopping until now)
    n_components = number of CCA outputs

    Returns
    ctw path
    -------

    """

    cca = CCA(n_components=n_components)

    # first iteration :
    # identity matrices, this relates to apply first a dtw on the inputs
    new_path, score_match = dtw_path(seq1, seq2,
                                     global_constraint=global_constraint,
                                     sakoe_chiba_radius=sakoe_chiba_radius,
                                     itakura_max_slope=itakura_max_slope)
    current_score = score_match
    current_path = np.asarray(new_path)

    if verbose:
        print("iteration ", 0, " : SCORE = ", current_score)

    for it in range(max_iter-1):
        Wx, Wy = _get_warp_matrices(current_path)

        cca.fit(Wx.dot(seq1), Wy.dot(seq2))
        seq1_tr, seq2_tr = cca.transform(seq1, seq2)

        new_path, score_match = dtw_path(seq1_tr, seq2_tr,
                                         global_constraint=global_constraint,
                                         sakoe_chiba_radius=sakoe_chiba_radius,
                                         itakura_max_slope=itakura_max_slope)

        if np.array_equal(current_path, new_path):
            break

        current_score = score_match
        current_path = np.asarray(new_path)

        if verbose:
            print("iteration ", it+1, " : SCORE = ", current_score)

    return current_path, cca, current_score
