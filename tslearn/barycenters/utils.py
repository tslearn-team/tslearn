import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'



def _set_weights(w, n):
    """Return w if it is a valid weight vector of size n, and a vector of n 1s
    otherwise.
    """
    if w is None or len(w) != n:
        w = numpy.ones((n, ))
    return w
