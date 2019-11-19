"""
The :mod:`tslearn.utils` module includes various utilities.
"""

import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d, check_array
from sklearn.utils.validation import check_is_fitted
import warnings

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def check_dims(X, X_fit=None, extend=True):
    """Reshapes X to a 3-dimensional array of X.shape[0] univariate
    timeseries of length X.shape[1] if X is 2-dimensional and extend
    is True. Then checks whether the dimensions, except the first one,
    of X_fit and X match.

    Parameters
    ----------
    X : array-like
        The first array to be compared.
    X_fit : array-like or None (default: None)
        The second array to be compared, which is created during fit.
        If None, then only perform reshaping of X, if necessary.
    extend : boolean (default: True)
        Whether to reshape X, if it is 2-dimensional.

    Returns
    -------
    array
        Reshaped X array

    Examples
    --------
    >>> X = numpy.empty((10, 3))
    >>> check_dims(X).shape
    (10, 3, 1)
    >>> X = numpy.empty((10, 3, 1))
    >>> check_dims(X).shape
    (10, 3, 1)
    >>> X_fit = numpy.empty((5, 3, 1))
    >>> check_dims(X, X_fit).shape
    (10, 3, 1)
    >>> X_fit = numpy.empty((5, 3, 2))
    >>> check_dims(X, X_fit)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Dimensions (except first) must match! ((5, 3, 2) and (10, 3, 1)
    are passed shapes)

    Raises
    ------
    ValueError
        Will raise exception if X is None or (if X_fit is provided) one of the
        dimensions, except the first, does not match.
    """
    if X is None:
        raise ValueError('X is equal to None!')

    if extend and len(X.shape) == 2:
        warnings.warn('2-Dimensional data passed. Assuming these are '
                      '{} 1-dimensional timeseries'.format(X.shape[0]))
        X = X.reshape((X.shape) + (1,))

    if X_fit is not None and X_fit.shape[1:] != X.shape[1:]:
        raise ValueError('Dimensions (except first) must match!'
                         ' ({} and {} are passed shapes)'.format(X_fit.shape,
                                                                 X.shape))

    return X


def _arraylike_copy(arr):
    """Duplicate content of arr into a numpy array.
     """
    if type(arr) != numpy.ndarray:
        return numpy.array(arr)
    else:
        return arr.copy()


def bit_length(n):
    """Returns the number of bits necessary to represent an integer in binary,
    excluding the sign and leading zeros.

    This function is provided for Python 2.6 compatibility.

    Examples
    --------
    >>> bit_length(0)
    0
    >>> bit_length(1)
    1
    >>> bit_length(2)
    2
    """
    k = 0
    try:
        if n > 0:
            k = n.bit_length()
    except AttributeError:  # In Python2.6, bit_length does not exist
        k = 1 + int(numpy.log2(abs(n)))
    return k


def to_time_series(ts, remove_nans=False):
    """Transforms a time series so that it fits the format used in ``tslearn``
    models.

    Parameters
    ----------
    ts : array-like
        The time series to be transformed.
    remove_nans : bool (default: False)
        Whether trailing NaNs at the end of the time series should be removed
        or not

    Returns
    -------
    numpy.ndarray of shape (sz, d)
        The transformed time series.

    Examples
    --------
    >>> to_time_series([1, 2])
    array([[1.],
           [2.]])
    >>> to_time_series([1, 2, numpy.nan])
    array([[ 1.],
           [ 2.],
           [nan]])
    >>> to_time_series([1, 2, numpy.nan], remove_nans=True)
    array([[1.],
           [2.]])

    See Also
    --------
    to_time_series_dataset : Transforms a dataset of time series
    """
    ts_out = _arraylike_copy(ts)
    if ts_out.ndim == 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != numpy.float:
        ts_out = ts_out.astype(numpy.float)
    if remove_nans:
        ts_out = ts_out[:ts_size(ts_out)]
    return ts_out


def to_time_series_dataset(dataset, dtype=numpy.float):
    """Transforms a time series dataset so that it fits the format used in
    ``tslearn`` models.

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

    Examples
    --------
    >>> to_time_series_dataset([[1, 2]])
    array([[[1.],
            [2.]]])
    >>> to_time_series_dataset([[1, 2], [1, 4, 3]])
    array([[[ 1.],
            [ 2.],
            [nan]],
    <BLANKLINE>
           [[ 1.],
            [ 4.],
            [ 3.]]])

    See Also
    --------
    to_time_series : Transforms a single time series
    """
    if len(dataset) == 0:
        return numpy.zeros((0, 0, 0))
    if numpy.array(dataset[0]).ndim == 0:
        dataset = [dataset]
    n_ts = len(dataset)
    max_sz = max([ts_size(to_time_series(ts)) for ts in dataset])
    d = to_time_series(dataset[0]).shape[1]
    dataset_out = numpy.zeros((n_ts, max_sz, d), dtype=dtype) + numpy.nan
    for i in range(n_ts):
        ts = to_time_series(dataset[i], remove_nans=True)
        dataset_out[i, :ts.shape[0]] = ts
    return dataset_out


def to_sklearn_dataset(dataset, dtype=numpy.float, return_dim=False):
    """Transforms a time series dataset so that it fits the format used in
    ``sklearn`` estimators.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed.
    dtype : data type (default: numpy.float)
        Data type for the returned dataset.
    return_dim : boolean  (optional, default: False)
        Whether the dimensionality (third dimension should be returned together
        with the transformed dataset).

    Returns
    -------
    numpy.ndarray of shape (n_ts, sz * d)
        The transformed dataset of time series.
    int (optional, if return_dim=True)
        The dimensionality of the original tslearn dataset (third dimension)

    Examples
    --------
    >>> to_sklearn_dataset([[1, 2]], return_dim=True)
    (array([[1., 2.]]), 1)
    >>> to_sklearn_dataset([[1, 2], [1, 4, 3]])
    array([[ 1.,  2., nan],
           [ 1.,  4.,  3.]])

    See Also
    --------
    to_time_series_dataset : Transforms a time series dataset to ``tslearn``
    format.
    """
    tslearn_dataset = to_time_series_dataset(dataset, dtype=dtype)
    n_ts = tslearn_dataset.shape[0]
    d = tslearn_dataset.shape[2]
    if return_dim:
        return tslearn_dataset.reshape((n_ts, -1)), d
    else:
        return tslearn_dataset.reshape((n_ts, -1))


def time_series_to_str(ts, fmt="%.18e"):
    """Transforms a time series to its representation as a string (used when
    saving time series to disk).

    Parameters
    ----------
    ts : array-like
        Time series to be represented.
    fmt : string (default: "%.18e")
        Format to be used to write each value.

    Returns
    -------
    string
        String representation of the time-series.

    Examples
    --------
    >>> time_series_to_str([1, 2, 3, 4], fmt="%.1f")
    '1.0 2.0 3.0 4.0'
    >>> time_series_to_str([[1, 3], [2, 4]], fmt="%.1f")
    '1.0 2.0|3.0 4.0'

    See Also
    --------
    load_time_series_txt : Load time series from disk
    str_to_time_series : Transform a string into a time series
    """
    ts_ = to_time_series(ts)
    dim = ts_.shape[1]
    s = ""
    for d in range(dim):
        s += " ".join([fmt % v for v in ts_[:, d]])
        if d < dim - 1:
            s += "|"
    return s


timeseries_to_str = time_series_to_str


def str_to_time_series(ts_str):
    """Reads a time series from its string representation (used when loading
    time series from disk).

    Parameters
    ----------
    ts_str : string
        String representation of the time-series.

    Returns
    -------
    numpy.ndarray
        Represented time-series.

    Examples
    --------
    >>> str_to_time_series("1 2 3 4")
    array([[1.],
           [2.],
           [3.],
           [4.]])
    >>> str_to_time_series("1 2|3 4")
    array([[1., 3.],
           [2., 4.]])

    See Also
    --------
    load_time_series_txt : Load time series from disk
    time_series_to_str : Transform a time series into a string
    """
    dimensions = ts_str.split("|")
    ts = [dim_str.split(" ") for dim_str in dimensions]
    return to_time_series(numpy.transpose(ts))


str_to_timeseries = str_to_time_series


def save_time_series_txt(fname, dataset, fmt="%.18e"):
    """Writes a time series dataset to disk.

    Parameters
    ----------
    fname : string
        Path to the file in which time series should be written.
    dataset : array-like
        The dataset of time series to be saved.
    fmt : string (default: "%.18e")
        Format to be used to write each value.

    Examples
    --------
    >>> dataset = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3]])
    >>> save_time_series_txt("tmp-tslearn-test.txt", dataset)

    See Also
    --------
    load_time_series_txt : Load time series from disk
    """
    fp = open(fname, "wt")
    for ts in dataset:
        fp.write(time_series_to_str(ts, fmt=fmt) + "\n")
    fp.close()


save_timeseries_txt = save_time_series_txt


def load_time_series_txt(fname):
    """Loads a time series dataset from disk.

    Parameters
    ----------
    fname : string
        Path to the file from which time series should be read.

    Returns
    -------
    numpy.ndarray or array of numpy.ndarray
        The dataset of time series.

    Examples
    --------
    >>> dataset = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3]])
    >>> save_time_series_txt("tmp-tslearn-test.txt", dataset)
    >>> reloaded_dataset = load_time_series_txt("tmp-tslearn-test.txt")

    See Also
    --------
    save_time_series_txt : Save time series to disk
    """
    dataset = []
    fp = open(fname, "rt")
    for row in fp.readlines():
        ts = str_to_time_series(row)
        dataset.append(ts)
    fp.close()
    return to_time_series_dataset(dataset)


load_timeseries_txt = load_time_series_txt


def check_equal_size(dataset):
    """Check if all time series in the dataset have the same size.

    Parameters
    ----------
    dataset: array-like
        The dataset to check.

    Returns
    -------
    bool
        Whether all time series in the dataset have the same size.

    Examples
    --------
    >>> check_equal_size([[1, 2, 3], [4, 5, 6], [5, 3, 2]])
    True
    >>> check_equal_size([[1, 2, 3, 4], [4, 5, 6], [5, 3, 2]])
    False
    """
    dataset_ = to_time_series_dataset(dataset)
    sz = -1
    for ts in dataset_:
        if sz < 0:
            sz = ts_size(ts)
        else:
            if sz != ts_size(ts):
                return False
    return True


def ts_size(ts):
    """Returns actual time series size.

    Final timesteps that have NaN values for all dimensions will be removed
    from the count.

    Parameters
    ----------
    ts : array-like
        A time series.

    Returns
    -------
    int
        Actual size of the time series.

    Examples
    --------
    >>> ts_size([1, 2, 3, numpy.nan])
    3
    >>> ts_size([1, numpy.nan])
    1
    >>> ts_size([numpy.nan])
    0
    >>> ts_size([[1, 2],
    ...          [2, 3],
    ...          [3, 4],
    ...          [numpy.nan, 2],
    ...          [numpy.nan, numpy.nan]])
    4
    """
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    while sz > 0 and not numpy.any(numpy.isfinite(ts_[sz - 1])):
        sz -= 1
    return sz


def ts_zeros(sz, d=1):
    """Returns a time series made of zero values.

    Parameters
    ----------
    sz : int
        Time series size.
    d : int (optional, default: 1)
        Time series dimensionality.

    Returns
    -------
    numpy.ndarray
        A time series made of zeros.

    Examples
    --------
    >>> ts_zeros(3, 2)  # doctest: +NORMALIZE_WHITESPACE
    array([[0., 0.],
           [0., 0.],
           [0., 0.]])
    >>> ts_zeros(5).shape
    (5, 1)
    """
    return numpy.zeros((sz, d))


def check_dataset(X, force_univariate=False, force_equal_length=False,
                  force_single_time_series=False):
    """Check if X is a valid tslearn dataset, with possibly additional extra
    constraints.

    Parameters
    ----------
    X: array, shape = (n_ts, sz, d)
        Time series dataset.
    force_univariate: bool (default: False)
        If True, only univariate datasets are considered valid.
    force_equal_length: bool (default: False)
        If True, only equal-length datasets are considered valid.
    force_single_time_series: bool (default: False)
        If True, only datasets made of a single time series are considered
        valid.

    Returns
    -------
    array, shape = (n_ts, sz, d)
        Formatted dataset, if it is valid

    Raises
    ------
    ValueError
        Raised if X is not a valid dataset, or one of the constraints is not
        satisfied.

    Examples
    --------
    >>> X = [[1, 2, 3], [1, 2, 3, 4]]
    >>> X_new = check_dataset(X)
    >>> X_new.shape
    (2, 4, 1)
    >>> check_dataset(
    ...     X,
    ...     force_equal_length=True
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: All the time series in the array should be of equal lengths.
    >>> other_X = numpy.random.randn(3, 10, 2)
    >>> check_dataset(
    ...     other_X,
    ...     force_univariate=True
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: Array should be univariate and is of shape: (3, 10, 2)
    >>> other_X = numpy.random.randn(3, 10, 2)
    >>> check_dataset(
    ...     other_X,
    ...     force_single_time_series=True
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: Array should be made of a single time series (3 here)
    """
    X_ = to_time_series_dataset(X)
    if force_univariate and X_.shape[2] != 1:
        raise ValueError(
            "Array should be univariate and is of shape: {}".format(
                X_.shape
            )
        )
    if force_equal_length and not check_equal_size(X_):
        raise ValueError("All the time series in the array should be of "
                         "equal lengths")
    if force_single_time_series and X_.shape[0] != 1:
        raise ValueError("Array should be made of a single time series "
                         "({} here)".format(X_.shape[0]))
    return X_


def to_pyts_dataset(X):
    """Transform a tslearn-compatible dataset into a pyts dataset.

    Parameters
    ----------
    X: array, shape = (n_ts, sz, d)
        tslearn-formatted dataset to be cast to pyts format

    Returns
    -------
    array, shape=(n_ts, sz) if d=1, (n_ts, d, sz) otherwise
        pyts-formatted dataset

    Examples
    --------
    >>> tslearn_arr = numpy.random.randn(10, 16, 1)
    >>> pyts_arr = to_pyts_dataset(tslearn_arr)
    >>> pyts_arr.shape
    (10, 16)
    >>> tslearn_arr = numpy.random.randn(10, 16, 2)
    >>> pyts_arr = to_pyts_dataset(tslearn_arr)
    >>> pyts_arr.shape
    (10, 2, 16)
    >>> tslearn_arr = [numpy.random.randn(16, 1), numpy.random.randn(10, 1)]
    >>> to_pyts_dataset(tslearn_arr)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: All the time series in the array should be of equal lengths
    """
    X_ = check_dataset(X, force_equal_length=True)
    if X_.shape[2] == 1:
        return X_.reshape((X_.shape[0], -1))
    else:
        return X_.transpose((0, 2, 1))


def from_pyts_dataset(X):
    """Transform a pyts-compatible dataset into a tslearn dataset.

    Parameters
    ----------
    X: array, shape = (n_ts, sz) or (n_ts, d, sz)
        pyts-formatted dataset

    Returns
    -------
    array, shape=(n_ts, sz, d)
        tslearn-formatted dataset

    Examples
    --------
    >>> pyts_arr = numpy.random.randn(10, 16)
    >>> tslearn_arr = from_pyts_dataset(pyts_arr)
    >>> tslearn_arr.shape
    (10, 16, 1)
    >>> pyts_arr = numpy.random.randn(10, 2, 16)
    >>> tslearn_arr = from_pyts_dataset(pyts_arr)
    >>> tslearn_arr.shape
    (10, 16, 2)
    >>> pyts_arr = numpy.random.randn(10)
    >>> from_pyts_dataset(pyts_arr)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: X is not a valid input pyts array.
    """
    X_ = check_array(X, ensure_2d=False, allow_nd=True)
    if X_.ndim == 2:
        shape = list(X_.shape) + [1]
        return X_.reshape(shape)
    elif X_.ndim == 3:
        return X_.transpose((0, 2, 1))
    else:
        raise ValueError("X is not a valid input pyts array. "
                         "Its dimensions, once cast to numpy.ndarray "
                         "are {}".format(X_.shape))


def to_seglearn_dataset(X):
    """Transform a tslearn-compatible dataset into a seglearn dataset.

    Parameters
    ----------
    X: array, shape = (n_ts, sz, d)
        tslearn-formatted dataset to be cast to seglearn format

    Returns
    -------
    array of arrays, shape=(n_ts, )
        seglearn-formatted dataset. i-th sub-array in the list has shape
        (sz_i, d)

    Examples
    --------
    >>> tslearn_arr = numpy.random.randn(10, 16, 1)
    >>> seglearn_arr = to_seglearn_dataset(tslearn_arr)
    >>> seglearn_arr.shape
    (10, 16, 1)
    >>> tslearn_arr = numpy.random.randn(10, 16, 2)
    >>> seglearn_arr = to_seglearn_dataset(tslearn_arr)
    >>> seglearn_arr.shape
    (10, 16, 2)
    >>> tslearn_arr = [numpy.random.randn(16, 2), numpy.random.randn(10, 2)]
    >>> seglearn_arr = to_seglearn_dataset(tslearn_arr)
    >>> seglearn_arr.shape
    (2,)
    >>> seglearn_arr[0].shape
    (16, 2)
    >>> seglearn_arr[1].shape
    (10, 2)
    """
    X_ = check_dataset(X)
    return numpy.array([Xi[:ts_size(Xi)] for Xi in X_])


def from_seglearn_dataset(X):
    """Transform a seglearn-compatible dataset into a tslearn dataset.

    Parameters
    ----------
    X: list of arrays, or array of arrays, shape = (n_ts, )
        seglearn-formatted dataset. i-th sub-array in the list has shape
        (sz_i, d)

    Returns
    -------
    array, shape=(n_ts, sz, d), where sz is the maximum of all array lengths
        tslearn-formatted dataset

    Examples
    --------
    >>> seglearn_arr = [numpy.random.randn(10, 1), numpy.random.randn(10, 1)]
    >>> tslearn_arr = from_seglearn_dataset(seglearn_arr)
    >>> tslearn_arr.shape
    (2, 10, 1)
    >>> seglearn_arr = [numpy.random.randn(10, 1), numpy.random.randn(5, 1)]
    >>> tslearn_arr = from_seglearn_dataset(seglearn_arr)
    >>> tslearn_arr.shape
    (2, 10, 1)
    >>> seglearn_arr = numpy.random.randn(2, 10, 1)
    >>> tslearn_arr = from_seglearn_dataset(seglearn_arr)
    >>> tslearn_arr.shape
    (2, 10, 1)
    """
    return to_time_series_dataset(X)


def to_stumpy_dataset(X):
    """Transform a tslearn-compatible dataset into a stumpy dataset.

    Parameters
    ----------
    X: array, shape = (n_ts, sz, d)
        tslearn-formatted dataset to be cast to stumpy format

    Returns
    -------
    list of arrays of shape=(d, sz_i) if d > 1 or (sz_i, ) otherwise
        stumpy-formatted dataset.

    Examples
    --------
    >>> tslearn_arr = numpy.random.randn(10, 16, 1)
    >>> stumpy_arr = to_stumpy_dataset(tslearn_arr)
    >>> len(stumpy_arr)
    10
    >>> stumpy_arr[0].shape
    (16,)
    >>> tslearn_arr = numpy.random.randn(10, 16, 2)
    >>> stumpy_arr = to_stumpy_dataset(tslearn_arr)
    >>> len(stumpy_arr)
    10
    >>> stumpy_arr[0].shape
    (2, 16)
    """
    X_ = check_dataset(X)

    def transpose_or_flatten(ts):
        if ts.shape[1] == 1:
            return ts.reshape((-1, ))
        else:
            return ts.transpose()

    return [transpose_or_flatten(Xi[:ts_size(Xi)]) for Xi in X_]


def from_stumpy_dataset(X):
    """Transform a stumpy-compatible dataset into a tslearn dataset.

    Parameters
    ----------
    X: list of arrays of shapes (d, sz_i) if d > 1 or (sz_i, ) otherwise
        stumpy-formatted dataset.

    Returns
    -------
    array, shape=(n_ts, sz, d), where sz is the maximum of all array lengths
        tslearn-formatted dataset

    Examples
    --------
    >>> stumpy_arr = [numpy.random.randn(10), numpy.random.randn(10)]
    >>> tslearn_arr = from_stumpy_dataset(stumpy_arr)
    >>> tslearn_arr.shape
    (2, 10, 1)
    >>> stumpy_arr = [numpy.random.randn(3, 10), numpy.random.randn(3, 5)]
    >>> tslearn_arr = from_stumpy_dataset(stumpy_arr)
    >>> tslearn_arr.shape
    (2, 10, 3)
    """
    def transpose_or_expand(ts):
        if ts.ndim == 1:
            return ts.reshape((-1, 1))
        else:
            return ts.transpose()
    return to_time_series_dataset([transpose_or_expand(Xi) for Xi in X])


try:  # Ugly hack, not sure how to to it better
    import pandas as pd

    def to_sktime_dataset(X):
        """Transform a tslearn-compatible dataset into a sktime dataset.

        Parameters
        ----------
        X: array, shape = (n_ts, sz, d)
            tslearn-formatted dataset to be cast to sktime format

        Returns
        -------
        Pandas data-frame
            sktime-formatted dataset (cf.
            `link <https://alan-turing-institute.github.io/sktime/examples/loading_data.html>`_)

        Examples
        --------
        >>> tslearn_arr = numpy.random.randn(10, 16, 1)
        >>> sktime_arr = to_sktime_dataset(tslearn_arr)
        >>> sktime_arr.shape
        (10, 1)
        >>> sktime_arr["dim_0"][0].shape
        (16,)
        >>> tslearn_arr = numpy.random.randn(10, 16, 2)
        >>> sktime_arr = to_sktime_dataset(tslearn_arr)
        >>> sktime_arr.shape
        (10, 2)
        >>> sktime_arr["dim_1"][0].shape
        (16,)

        Notes
        -----
        Conversion from/to sktime format requires pandas to be installed.
        """  # noqa: E501
        X_ = check_dataset(X)
        X_pd = pd.DataFrame(dtype=numpy.float32)
        for dim in range(X_.shape[2]):
            X_pd['dim_' + str(dim)] = [pd.Series(data=Xi[:ts_size(Xi), dim])
                                       for Xi in X_]
        return X_pd

    def from_sktime_dataset(X):
        """Transform a sktime-compatible dataset into a tslearn dataset.

        Parameters
        ----------
        X: pandas data-frame
            sktime-formatted dataset (cf.
            `link <https://alan-turing-institute.github.io/sktime/examples/loading_data.html>`_)

        Returns
        -------
        array, shape=(n_ts, sz, d)
            tslearn-formatted dataset

        Examples
        --------
        >>> sktime_df = pd.DataFrame()
        >>> sktime_df["dim_0"] = [pd.Series([1, 2, 3]), pd.Series([4, 5, 6])]
        >>> tslearn_arr = from_sktime_dataset(sktime_df)
        >>> tslearn_arr.shape
        (2, 3, 1)
        >>> sktime_df = pd.DataFrame()
        >>> sktime_df["dim_0"] = [pd.Series([1, 2, 3]),
        ...                       pd.Series([4, 5, 6, 7])]
        >>> sktime_df["dim_1"] = [pd.Series([8, 9, 10]),
        ...                       pd.Series([11, 12, 13, 14])]
        >>> tslearn_arr = from_sktime_dataset(sktime_df)
        >>> tslearn_arr.shape
        (2, 4, 2)
        >>> sktime_arr = numpy.random.randn(10, 1, 16)
        >>> from_sktime_dataset(
        ...     sktime_arr
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: X is not a valid input sktime array.

        Notes
        -----
        Conversion from/to sktime format requires pandas to be installed.
        """  # noqa: E501
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X is not a valid input sktime array. "
                             "A pandas DataFrame is expected.")
        data_dimensions = [col_name
                           for col_name in X.columns
                           if col_name.startswith("dim_")]
        d = len(data_dimensions)
        ordered_data_dimensions = ["dim_%d" % di for di in range(d)]
        if sorted(ordered_data_dimensions) != sorted(data_dimensions):
            raise ValueError("X is not a valid input sktime array. "
                             "Provided dimensions are not conitiguous."
                             "{}".format(data_dimensions))
        n = X["dim_0"].shape[0]
        max_sz = -1
        for dim_name in ordered_data_dimensions:
            for i in range(n):
                if X[dim_name][i].size > max_sz:
                    max_sz = X[dim_name][i].size

        tslearn_arr = numpy.empty((n, max_sz, d))
        tslearn_arr[:] = numpy.nan
        for di in range(d):
            for i in range(n):
                sz = X["dim_%d" % di][i].size
                tslearn_arr[i, :sz, di] = X["dim_%d" % di][i].values.copy()
        return tslearn_arr

    def to_pyflux_dataset(X):
        """Transform a tslearn-compatible dataset into a pyflux dataset.

        Parameters
        ----------
        X: array, shape = (n_ts, sz, d), where n_ts=1
            tslearn-formatted dataset to be cast to pyflux format

        Returns
        -------
        Pandas data-frame
            pyflux-formatted dataset (cf.
            `link <https://pyflux.readthedocs.io/en/latest/getting_started.html>`_)

        Examples
        --------
        >>> tslearn_arr = numpy.random.randn(1, 16, 1)
        >>> pyflux_df = to_pyflux_dataset(tslearn_arr)
        >>> pyflux_df.shape
        (16, 1)
        >>> pyflux_df.columns[0]
        'dim_0'
        >>> tslearn_arr = numpy.random.randn(1, 16, 2)
        >>> pyflux_df = to_pyflux_dataset(tslearn_arr)
        >>> pyflux_df.shape
        (16, 2)
        >>> pyflux_df.columns[1]
        'dim_1'
        >>> tslearn_arr = numpy.random.randn(10, 16, 1)
        >>> to_pyflux_dataset(tslearn_arr)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Array should be made of a single time series (10 here)

        Notes
        -----
        Conversion from/to pyflux format requires pandas to be installed.
        """  # noqa: E501
        X_ = check_dataset(X,
                           force_equal_length=True,
                           force_single_time_series=True)
        X_pd = pd.DataFrame(X[0], dtype=numpy.float32)
        X_pd.columns = ["dim_%d" % di for di in range(X_.shape[2])]
        return X_pd

    def from_pyflux_dataset(X):
        """Transform a pyflux-compatible dataset into a tslearn dataset.

        Parameters
        ----------
        X: pandas data-frame
            pyflux-formatted dataset

        Returns
        -------
        array, shape=(n_ts, sz, d), where n_ts=1
            tslearn-formatted dataset.
            Column order is kept the same as in the original data frame.

        Examples
        --------
        >>> pyflux_df = pd.DataFrame()
        >>> pyflux_df["dim_0"] = numpy.random.rand(10)
        >>> tslearn_arr = from_pyflux_dataset(pyflux_df)
        >>> tslearn_arr.shape
        (1, 10, 1)
        >>> pyflux_df = pd.DataFrame()
        >>> pyflux_df["dim_0"] = numpy.random.rand(10)
        >>> pyflux_df["dim_1"] = numpy.random.rand(10)
        >>> pyflux_df["dim_2"] = numpy.random.rand(10)
        >>> tslearn_arr = from_pyflux_dataset(pyflux_df)
        >>> tslearn_arr.shape
        (1, 10, 3)
        >>> pyflux_arr = numpy.random.randn(10, 1, 16)
        >>> from_pyflux_dataset(
        ...     pyflux_arr
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: X is not a valid input pyflux array.

        Notes
        -----
        Conversion from/to pyflux format requires pandas to be installed.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X is not a valid input pyflux array. "
                             "A pandas DataFrame is expected.")
        data_dimensions = [col_name for col_name in X.columns]
        d = len(data_dimensions)
        n = 1

        max_sz = -1
        for dim_name in data_dimensions:
            if X[dim_name].size > max_sz:
                max_sz = X[dim_name].size

        tslearn_arr = numpy.empty((n, max_sz, d))
        tslearn_arr[:] = numpy.nan
        for di, dim_name in enumerate(data_dimensions):
            data = X[dim_name].values.copy()
            sz = len(data)
            tslearn_arr[0, :sz, di] = data
        return tslearn_arr

    def to_tsfresh_dataset(X):
        """Transform a tslearn-compatible dataset into a tsfresh dataset.

        Parameters
        ----------
        X: array, shape = (n_ts, sz, d)
            tslearn-formatted dataset to be cast to tsfresh format

        Returns
        -------
        Pandas data-frame
            tsfresh-formatted dataset ("flat" data frame, as described
            `there <https://tsfresh.readthedocs.io/en/latest/text/data_formats.html#input-option-1-flat-dataframe>`_)

        Examples
        --------
        >>> tslearn_arr = numpy.random.randn(1, 16, 1)
        >>> tsfresh_df = to_tsfresh_dataset(tslearn_arr)
        >>> tsfresh_df.shape
        (16, 3)
        >>> tslearn_arr = numpy.random.randn(1, 16, 2)
        >>> tsfresh_df = to_tsfresh_dataset(tslearn_arr)
        >>> tsfresh_df.shape
        (16, 4)

        Notes
        -----
        Conversion from/to tsfresh format requires pandas to be installed.
        """  # noqa: E501
        X_ = check_dataset(X)
        n, sz, d = X_.shape
        dataframes = []
        for i, Xi in enumerate(X_):
            df = pd.DataFrame(columns=["id", "time"] +
                                      ["dim_%d" % di for di in range(d)])
            Xi_ = Xi[:ts_size(Xi)]
            sz = Xi_.shape[0]
            df["time"] = numpy.arange(sz)
            df["id"] = numpy.zeros((sz, ), dtype=numpy.int32) + i
            for di in range(d):
                df["dim_%d" % di] = Xi_[:, di]
            dataframes.append(df)
        return pd.concat(dataframes)

    def from_tsfresh_dataset(X):
        """Transform a tsfresh-compatible dataset into a tslearn dataset.

        Parameters
        ----------
        X: pandas data-frame
            tsfresh-formatted dataset ("flat" data frame, as described
            `there <https://tsfresh.readthedocs.io/en/latest/text/data_formats.html#input-option-1-flat-dataframe>`_)

        Returns
        -------
        array, shape=(n_ts, sz, d)
            tslearn-formatted dataset.
            Column order is kept the same as in the original data frame.

        Examples
        --------
        >>> tsfresh_df = pd.DataFrame(columns=["id", "time", "a", "b"])
        >>> tsfresh_df["id"] = [0, 0, 0]
        >>> tsfresh_df["time"] = [0, 1, 2]
        >>> tsfresh_df["a"] = [-1, 4, 7]
        >>> tsfresh_df["b"] = [8, -3, 2]
        >>> tslearn_arr = from_tsfresh_dataset(tsfresh_df)
        >>> tslearn_arr.shape
        (1, 3, 2)
        >>> tsfresh_df = pd.DataFrame(columns=["id", "time", "a"])
        >>> tsfresh_df["id"] = [0, 0, 0, 1, 1]
        >>> tsfresh_df["time"] = [0, 1, 2, 0, 1]
        >>> tsfresh_df["a"] = [-1, 4, 7, 9, 1]
        >>> tslearn_arr = from_tsfresh_dataset(tsfresh_df)
        >>> tslearn_arr.shape
        (2, 3, 1)
        >>> tsfresh_df = numpy.random.randn(10, 1, 16)
        >>> from_tsfresh_dataset(
        ...     tsfresh_df
        ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: X is not a valid input tsfresh array.

        Notes
        -----
        Conversion from/to tsfresh format requires pandas to be installed.
        """  # noqa: E501
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X is not a valid input tsfresh array. "
                             "A pandas DataFrame is expected.")
        data_dimensions = [col_name
                           for col_name in X.columns
                           if col_name not in ["id", "time"]]
        d = len(data_dimensions)
        all_ids = set(X["id"])
        n = len(all_ids)

        max_sz = -1
        for ind_id in all_ids:
            sz = X[X["id"] == ind_id].shape[0]
            if sz > max_sz:
                max_sz = sz

        tslearn_arr = numpy.empty((n, max_sz, d))
        tslearn_arr[:] = numpy.nan
        for di, dim_name in enumerate(data_dimensions):
            for i, ind_id in enumerate(all_ids):
                data_ind = X[X["id"] == ind_id]
                data = data_ind[dim_name]
                sz = data_ind.shape[0]
                tslearn_arr[i, :sz, di] = data
        return tslearn_arr

except ImportError:
    def to_pyflux_dataset(X):
        raise ImportWarning("Conversion from/to pyflux cannot be performed "
                            "if pandas is not installed.")

    def from_pyflux_dataset(X):
        raise ImportWarning("Conversion from/to pyflux cannot be performed "
                            "if pandas is not installed.")

    def to_sktime_dataset(X):
        raise ImportWarning("Conversion from/to sktime cannot be performed "
                            "if pandas is not installed.")

    def from_sktime_dataset(X):
        raise ImportWarning("Conversion from/to sktime cannot be performed "
                            "if pandas is not installed.")

try:
    from cesium.time_series import TimeSeries

    def to_cesium_dataset(X):
        """Transform a tslearn-compatible dataset into a cesium dataset.

        Parameters
        ----------
        X: array, shape = (n_ts, sz, d), where n_ts=1
            tslearn-formatted dataset to be cast to cesium format

        Returns
        -------
        list of cesium TimeSeries
            cesium-formatted dataset (cf.
            `link <http://cesium-ml.org/docs/api/cesium.time_series.html#cesium.time_series.TimeSeries>`_)

        Examples
        --------
        >>> tslearn_arr = numpy.random.randn(3, 16, 1)
        >>> cesium_ds = to_cesium_dataset(tslearn_arr)
        >>> len(cesium_ds)
        3
        >>> cesium_ds[0].measurement.shape
        (16,)
        >>> tslearn_arr = numpy.random.randn(3, 16, 2)
        >>> cesium_ds = to_cesium_dataset(tslearn_arr)
        >>> len(cesium_ds)
        3
        >>> cesium_ds[0].measurement.shape
        (2, 16)
        >>> tslearn_arr = [[1, 2, 3], [1, 2, 3, 4]]
        >>> cesium_ds = to_cesium_dataset(tslearn_arr)
        >>> len(cesium_ds)
        2
        >>> cesium_ds[0].measurement.shape
        (3,)

        Notes
        -----
        Conversion from/to cesium format requires cesium to be installed.
        """  # noqa: E501
        def transpose_or_flatten(ts):
            ts_ = ts[:ts_size(ts)]
            if ts.shape[1] == 1:
                return ts_.reshape((-1, ))
            else:
                return ts_.transpose()

        X_ = check_dataset(X)
        return [TimeSeries(m=transpose_or_flatten(Xi)) for Xi in X_]

    def from_cesium_dataset(X):
        """Transform a cesium-compatible dataset into a tslearn dataset.

        Parameters
        ----------
        X: list of cesium TimeSeries
            cesium-formatted dataset (cf.
            `link <http://cesium-ml.org/docs/api/cesium.time_series.html#cesium.time_series.TimeSeries>`_)

        Returns
        -------
        array, shape=(n_ts, sz, d)
            tslearn-formatted dataset.

        Examples
        --------
        >>> cesium_ds = [TimeSeries(m=numpy.array([1, 2, 3, 4]))]
        >>> tslearn_arr = from_cesium_dataset(cesium_ds)
        >>> tslearn_arr.shape
        (1, 4, 1)
        >>> cesium_ds = [
        ...     TimeSeries(m=numpy.array([[1, 2, 3, 4],
        ...                               [5, 6, 7, 8]]))
        ... ]
        >>> tslearn_arr = from_cesium_dataset(cesium_ds)
        >>> tslearn_arr.shape
        (1, 4, 2)

        Notes
        -----
        Conversion from/to cesium format requires cesium to be installed.
        """  # noqa: E501
        def format_to_tslearn(ts):
            try:
                ts.sort()
            except ValueError:
                warnings.warn("Cesium dataset could not be sorted, assuming "
                              "it is already sorted before casting to "
                              "tslearn format.")
            if ts.measurement.ndim == 1:
                data = ts.measurement.reshape((1, -1))
            else:
                data = ts.measurement
            d = len(data)
            max_sz = max([len(ts_di) for ts_di in data])
            tslearn_ts = numpy.empty((max_sz, d))
            tslearn_ts[:] = numpy.nan
            for di in range(d):
                sz = data[di].shape[0]
                tslearn_ts[:sz, di] = data[di]
            return tslearn_ts

        if not isinstance(X, list) or \
                [type(ts) for ts in X] != [TimeSeries] * len(X):
            raise ValueError("X is not a valid input cesium array. "
                             "A list of cesium TimeSeries is expected.")
        dataset = [format_to_tslearn(ts) for ts in X]
        return to_time_series_dataset(dataset=dataset)
except ImportError:
    def to_cesium_dataset(X):
        raise ImportWarning("Conversion from/to cesium cannot be performed "
                            "if cesium is not installed.")

    def from_cesium_dataset(X):
        raise ImportWarning("Conversion from/to cesium cannot be performed "
                            "if cesium is not installed.")


class LabelCategorizer(BaseEstimator, TransformerMixin):
    """Transformer to transform indicator-based labels into categorical ones.

    Attributes
    ----------
    single_column_if_binary : boolean (optional, default: False)
        If true, generate a single column for binary classification case.
        Otherwise, will generate 2.
        If there are more than 2 labels, thie option will not change anything.
    forward_match : dict
        A dictionary that maps each element that occurs in the label vector
        on a index {y_i : i} with i in [0, C - 1], C the total number of
        unique labels and y_i the ith unique label.
    backward_match : array-like
        An array that maps an index back to the original label. Where
        backward_match[i] results in y_i.

    Examples
    --------
    >>> y = numpy.array([-1, 2, 1, 1, 2])
    >>> lc = LabelCategorizer()
    >>> lc.fit_transform(y)
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> lc.inverse_transform([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    array([ 1.,  2., -1.])
    >>> y = numpy.array([-1, 2, -1, -1, 2])
    >>> lc = LabelCategorizer(single_column_if_binary=True)
    >>> lc.fit_transform(y)
    array([[1.],
           [0.],
           [1.],
           [1.],
           [0.]])
    >>> lc.inverse_transform(lc.transform(y))
    array([-1.,  2., -1., -1.,  2.])

    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    def __init__(self, single_column_if_binary=False, forward_match=None,
                 backward_match=None):
        self.single_column_if_binary = single_column_if_binary
        self.forward_match = forward_match
        self.backward_match = backward_match

    def _init(self):
        self.forward_match = {}
        self.backward_match = []

    def fit(self, y):
        self._init()
        y = column_or_1d(y, warn=True)
        values = sorted(set(y))
        for i, v in enumerate(values):
            self.forward_match[v] = i
            self.backward_match.append(v)
        return self

    def transform(self, y):
        check_is_fitted(self, ['backward_match', 'forward_match'])
        y = column_or_1d(y, warn=True)
        n_classes = len(self.backward_match)
        n = len(y)
        y_out = numpy.zeros((n, n_classes))
        for i in range(n):
            y_out[i, self.forward_match[y[i]]] = 1
        if n_classes == 2 and self.single_column_if_binary:
            return y_out[:, 0].reshape((-1, 1))
        else:
            return y_out

    def inverse_transform(self, y):
        check_is_fitted(self, ['backward_match', 'forward_match'])
        y_ = numpy.array(y)
        n, n_c = y_.shape
        if n_c == 1 and self.single_column_if_binary:
            y_ = numpy.hstack((y_, 1 - y_))
        y_out = numpy.zeros((n, ))
        for i in range(n):
            y_out[i] = self.backward_match[y_[i].argmax()]
        return y_out

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = BaseEstimator.get_params(self, deep=deep)
        out["single_column_if_binary"] = self.single_column_if_binary
        out["forward_match"] = self.forward_match
        out["backward_match"] = self.backward_match
        return out

    def _get_tags(self):
        return {'X_types': ['1dlabels']}
