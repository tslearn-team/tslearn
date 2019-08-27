"""
The :mod:`tslearn.utils` module includes various utilities.
"""

import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
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


def timeseries_to_str(ts, fmt="%.18e"):
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
    >>> timeseries_to_str([1, 2, 3, 4], fmt="%.1f")
    '1.0 2.0 3.0 4.0'
    >>> timeseries_to_str([[1, 3], [2, 4]], fmt="%.1f")
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
        s += " ".join([fmt % v for v in ts_[:, d]])
        if d < dim - 1:
            s += "|"
    return s


def str_to_timeseries(ts_str):
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
    >>> str_to_timeseries("1 2 3 4")
    array([[1.],
           [2.],
           [3.],
           [4.]])
    >>> str_to_timeseries("1 2|3 4")
    array([[1., 3.],
           [2., 4.]])

    See Also
    --------
    load_timeseries_txt : Load time series from disk
    timeseries_to_str : Transform a time series into a string
    """
    dimensions = ts_str.split("|")
    ts = [dim_str.split(" ") for dim_str in dimensions]
    return to_time_series(numpy.transpose(ts))


def save_timeseries_txt(fname, dataset, fmt="%.18e"):
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
    >>> save_timeseries_txt("tmp-tslearn-test.txt", dataset)

    See Also
    --------
    load_timeseries_txt : Load time series from disk
    """
    fp = open(fname, "wt")
    for ts in dataset:
        fp.write(timeseries_to_str(ts, fmt=fmt) + "\n")
    fp.close()


def load_timeseries_txt(fname):
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
    >>> save_timeseries_txt("tmp-tslearn-test.txt", dataset)
    >>> reloaded_dataset = load_timeseries_txt("tmp-tslearn-test.txt")

    See Also
    --------
    save_timeseries_txt : Save time series to disk
    """
    dataset = []
    fp = open(fname, "rt")
    for row in fp.readlines():
        ts = str_to_timeseries(row)
        dataset.append(ts)
    fp.close()
    return to_time_series_dataset(dataset)


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
