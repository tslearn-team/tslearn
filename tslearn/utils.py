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


def bit_length(n):
    """Returns the number of bits necessary to represent an integer in binary, excluding the sign and leading zeros.

    This function is provided for Python 2.6 compatibility.

    Examples
    --------
    >>> bit_length(0)
    0
    >>> bit_length(2)
    2
    >>> bit_length(1)
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


def timeseries_to_str(ts):
    """Transforms a time series to its representation as a string (used when saving time series to disk).

    Parameters
    ----------
    ts : array-like
        Time series to be represented.

    Returns
    -------
    string
        String representation of the time-series.

    Examples
    --------
    >>> timeseries_to_str([1, 2, 3, 4])  # doctest: +NORMALIZE_WHITESPACE
    '1.0 2.0 3.0 4.0'
    >>> timeseries_to_str([[1, 3], [2, 4]])  # doctest: +NORMALIZE_WHITESPACE
    '1.0 2.0|3.0 4.0'

    See Also
    --------
    load_timeseries_txt : Load time series from disk
    str_to_timeseries : Transform a string into a time series
    """
    ts_ = to_time_series(ts)
    dim = ts_.shape[1]
    s = ""
    for d in range(dim):
        s += " ".join([str(v) for v in ts_[:, d]])
        if d < dim - 1:
            s += "|"
    return s


def str_to_timeseries(ts_str):
    """Transforms a time series to its representation as a string (used when saving time series to disk).

    Parameters
    ----------
    ts_str : string
        String representation of the time-series.

    Returns
    -------
    array
        Represented time-series.

    Examples
    --------
    >>> str_to_timeseries("1 2 3 4")  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.],
           [ 2.],
           [ 3.],
           [ 4.]])
    >>> str_to_timeseries("1 2|3 4")  # doctest: +NORMALIZE_WHITESPACE
    array([[ 1., 3.],
           [ 2., 4.]])

    See Also
    --------
    load_timeseries_txt : Load time series from disk
    timeseries_to_str : Transform a time series into a string
    """
    dimensions = ts_str.split("|")
    ts = [dim_str.split(" ") for dim_str in dimensions]
    return to_time_series(numpy.transpose(ts))


def save_timeseries_txt(fname, dataset):
    """Writes a time series dataset to disk.

    Parameters
    ----------
    fname : string
        Path to the file in which time setries should be written.
    dataset : array-like
        The dataset of time series to be saved.

    See Also
    --------
    load_timeseries_txt : Load time series from disk
    """
    fp = open(fname, "wt")
    for ts in dataset:
        fp.write(timeseries_to_str(ts) + "\n")
    fp.close()


def load_timeseries_txt(fname):
    """Loads a time series dataset from disk.

    Parameters
    ----------
    fname : string
        Path to the file from which time setries should be read.

    Returns
    -------
    array or list of arrays
        The dataset of time series.

    See Also
    --------
    save_timeseries_txt : Save time series to disk
    """
    dataset = []
    sz = -1
    equal_size = True
    fp = open(fname, "rt")
    for row in fp.readlines():
        ts = str_to_timeseries(row)
        if sz >= 0 and ts.shape[0] != sz:
            equal_size = False
        sz = ts.shape[0]
        dataset.append(ts)
    fp.close()
    if equal_size:
        return to_time_series_dataset(dataset, equal_size=True)
    else:
        return dataset
