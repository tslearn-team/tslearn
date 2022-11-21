"""Simple functions representing the tslearn functions."""

from backend.backend import Backend


def add(x, y):
    return x + y


def exp(x):
    be = Backend(x)
    return be.exp(x)


def log(x):
    be = Backend(x)
    return be.log(x)


def _inv_matrix_aux(x, be=None):
    if be is None:
        be = Backend(x)
    return be.linalg.inv(x)


def inv_matrices_main(x):
    be = Backend(x)
    inv_matrices = be.zeros(be.shape(x), x.dtype)
    for i_matrix in range(x.shape[0]):
        inv_matrices[i_matrix] = _inv_matrix_aux(x[i_matrix], be=be)
    return inv_matrices
