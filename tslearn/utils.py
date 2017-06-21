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


def npy2d_time_series(ts):
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
    >>> npy2d_time_series([1, 2]) # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.],
           [ 2.]])
    
    See Also
    --------
    npy3d_time_series_dataset : Transforms a dataset of time series
    """
    ts_out = _arraylike_copy(ts)
    if ts_out.ndim == 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != numpy.float:
        ts_out = ts_out.astype(numpy.float)
    return ts_out


def npy3d_time_series_dataset(dataset, dtype=numpy.float):
    """Transforms a time series dataset so that it fits the format used in ``tslearn`` models.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed.
    dtype : data type (default: numpy.float)
        Data type for the returned dataset.

    Returns
    -------
    numpy.ndarray of shape (n_ts, sz, d)
        The transformed dataset of time series.
    
    Example
    -------
    >>> npy3d_time_series_dataset([[1, 2]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1.],
            [ 2.]]])
    
    See Also
    --------
    npy2d_time_series : Transforms a single time series
    """
    dataset_out = _arraylike_copy(dataset)
    if dataset_out.ndim == 2:
        dataset_out = dataset_out.reshape((dataset_out.shape[0], dataset_out.shape[1], 1))
    if dataset_out.dtype != dtype:
        dataset_out = dataset_out.astype(dtype)
    return dataset_out
