"""
The :mod:`tslearn.utils` module includes various utilities.
"""

import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _arraylike_copy(arr):
    """Duplicate content of arr into a numpy array.

     Examples
     --------
     >>> X_npy = numpy.array([1, 2, 3])
     >>> numpy.alltrue(_arraylike_copy(X_npy) == X_npy)
     True
     >>> _arraylike_copy(X_npy) is X_npy
     False
     >>> numpy.alltrue(_arraylike_copy([1, 2, 3]) == X_npy)
     True
     """
    if type(arr) != numpy.ndarray:
        return numpy.array(arr)
    else:
        return arr.copy()


def _bit_length(n):
    """Returns the number of bits necessary to represent an integer in binary, excluding the sign and leading zeros.

    This function is provided for Python 2.6 compatibility.

    Examples
    --------
    >>> _bit_length(0)
    0
    >>> _bit_length(2)
    2
    >>> _bit_length(1)
    1
    """
    k = 0
    try:
        if n > 0:
            k = n.bit_length()
    except AttributeError:  # In Python2.6, bit_length does not exist
        k = 1 + int(numpy.log2(abs(n)))
    return k


def to_time_series(ts):
    """Transforms a time series so that it fits the format used in `tslearn` models.

    Parameters
    ----------
    ts : array-like
        The time series to be transformed

    Returns
    -------
    numpy.ndarray of shape (sz, d)
        The transformed time series
    
    Example
    -------
    >>> to_time_series([1, 2]) # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.],
           [ 2.]])
    
    See Also
    --------
    to_time_series_dataset : Transforms a dataset of time series
    """
    ts_out = _arraylike_copy(ts)
    if ts_out.ndim == 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != numpy.float:
        ts_out = ts_out.astype(numpy.float)
    return ts_out


def to_time_series_dataset(dataset, dtype=numpy.float, equal_size=True):
    """Transforms a time series dataset so that it fits the format used in ``tslearn`` models.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed.
    dtype : data type (default: numpy.float)
        Data type for the returned dataset.
    equal_size : bool (default: True)
        Whether generated time series are all supposed to be of the same size.

    Returns
    -------
    numpy.ndarray of shape (n_ts, sz, d) or list of numpy.ndarray of shape (sz_i, d)
        The transformed dataset of time series. Represented as a list of numpy arrays if equal_size is False.
    
    Example
    -------
    >>> to_time_series_dataset([[1, 2]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1.],
            [ 2.]]])
    >>> to_time_series_dataset([[1, 2], [1, 4, 3]], equal_size=False) # doctest: +NORMALIZE_WHITESPACE
    [array([[ 1.],
           [ 2.]]), array([[ 1.],
           [ 4.],
           [ 3.]])]
    
    See Also
    --------
    to_time_series : Transforms a single time series
    """
    if not equal_size:
        dataset_out = [to_time_series(ts) for ts in dataset]
    else:
        dataset_out = _arraylike_copy(dataset)
        if dataset_out.ndim == 1:
            dataset_out = dataset_out.reshape((1, dataset_out.shape[0], 1))
        elif dataset_out.ndim == 2:
            dataset_out = dataset_out.reshape((dataset_out.shape[0], dataset_out.shape[1], 1))
        if dataset_out.dtype != dtype:
            dataset_out = dataset_out.astype(dtype)
    return dataset_out
