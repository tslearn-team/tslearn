import numpy

from ..utils import to_time_series_dataset
from .utils import _set_weights

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

def euclidean_barycenter(X, weights=None):
    """Standard Euclidean barycenter computed from a set of time series.

    Parameters
    ----------
    X : array-like, shape=(n_ts, sz, d)
        Time series dataset.

    weights: None or array
        Weights of each X[i]. Must be the same size as len(X).
        If None, uniform weights are used.

    Returns
    -------
    numpy.array of shape (sz, d)
        Barycenter of the provided time series dataset.

    Notes
    -----
        This method requires a dataset of equal-sized time series

    Examples
    --------
    >>> time_series = [[1, 2, 3, 4], [1, 2, 4, 5]]
    >>> bar = euclidean_barycenter(time_series)
    >>> bar.shape
    (4, 1)
    >>> bar
    array([[1. ],
           [2. ],
           [3.5],
           [4.5]])
    """
    X_ = to_time_series_dataset(X)
    weights = _set_weights(weights, X_.shape[0])
    return numpy.average(X_, axis=0, weights=weights)
