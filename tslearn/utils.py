import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


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
    if type(ts) != numpy.ndarray:
        ts_out = numpy.array(ts)
    else:
        ts_out = ts.copy()
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
    if type(dataset) != numpy.ndarray:
        dataset_out = numpy.array(dataset)
    else:
        dataset_out = dataset.copy()
    if dataset_out.ndim == 2:
        dataset_out = dataset_out.reshape((dataset_out.shape[0], dataset_out.shape[1], 1))
    if dataset_out.dtype != dtype:
        dataset_out = dataset_out.astype(dtype)
    return dataset_out
