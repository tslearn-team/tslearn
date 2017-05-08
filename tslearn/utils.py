import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def npy2d_time_series(ts):
    """Transforms a time series so that it fits the format used in `tslearn` 
    models.
    
    :param ts: The time series to be transformed
    :return: The transformed time series (of shape ``(sz, d)``)
    :rtype: ``numpy.ndarray``
    
    :Example:
    
    >>> npy2d_time_series([1, 2]) # doctest: +NORMALIZE_WHITESPACE
    array([[ 1.],
           [ 2.]])
    
    .. seealso:: npy3d_time_series_dataset()
    """
    if type(ts) != numpy.ndarray:
        ts = numpy.array(ts)
    if ts.ndim == 1:
        ts = ts.reshape((-1, 1))
    if ts.dtype != numpy.float:
        ts = ts.astype(numpy.float)
    return ts


def npy3d_time_series_dataset(dataset):
    """Transforms a time series dataset so that it fits the format used in 
    ``tslearn`` models.
    
    :param dataset: The dataset of time series to be transformed
    :return: The transformed dataset of time series (of shape ``(n_ts, sz, d)``)
    :rtype: ``numpy.ndarray``
    
    :Example:
    
    >>> npy3d_time_series_dataset([[1, 2]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1.],
            [ 2.]]])
    
    .. seealso:: npy2d_time_series()
    """
    if type(dataset) != numpy.ndarray:
        dataset = numpy.array(dataset)
    if dataset.ndim == 2:
        dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], 1))
    if dataset.dtype != numpy.float:
        dataset = dataset.astype(numpy.float)
    return dataset


if __name__ == "__main__":
    import doctest
    doctest.testmod()
