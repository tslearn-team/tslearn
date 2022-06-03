import warnings
from io import StringIO

import numpy
from sklearn.base import TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted

try:
    from scipy.io import arff
    HAS_ARFF = True
except:
    HAS_ARFF = False

try:
    from sklearn.utils.estimator_checks import _NotAnArray as NotAnArray
except ImportError:  # Old sklearn versions
    from sklearn.utils.estimator_checks import NotAnArray
from tslearn.bases import TimeSeriesBaseEstimator

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def check_dims(X, X_fit_dims=None, extend=True, check_n_features_only=False):
    """Reshapes X to a 3-dimensional array of X.shape[0] univariate
    timeseries of length X.shape[1] if X is 2-dimensional and extend
    is True. Then checks whether the provided X_fit_dims and the
    dimensions of X (except for the first one), match.

    Parameters
    ----------
    X : array-like
        The first array to be compared.
    X_fit_dims : tuple (default: None)
        The dimensions of the data generated by fit, to compare with
        the dimensions of the provided array X.
        If None, then only perform reshaping of X, if necessary.
    extend : boolean (default: True)
        Whether to reshape X, if it is 2-dimensional.
    check_n_features_only: boolean (default: False)

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
    >>> X_fit_dims = (5, 3, 1)
    >>> check_dims(X, X_fit_dims).shape
    (10, 3, 1)
    >>> X_fit_dims = (5, 3, 2)
    >>> check_dims(X, X_fit_dims)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Dimensions (except first) must match! ((5, 3, 2) and (10, 3, 1)
    are passed shapes)
    >>> X_fit_dims = (5, 5, 1)
    >>> check_dims(X, X_fit_dims, check_n_features_only=True).shape
    (10, 3, 1)
    >>> X_fit_dims = (5, 5, 2)
    >>> check_dims(
    ...     X,
    ...     X_fit_dims,
    ...     check_n_features_only=True
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: Number of features of the provided timeseries must match!
    (last dimension) must match the one of the fitted data!
    ((5, 5, 2) and (10, 3, 1) are passed shapes)

    Raises
    ------
    ValueError
        Will raise exception if X is None or (if X_fit_dims is provided) one
        of the dimensions of the provided data, except the first, does not
        match X_fit_dims.
    """
    if X is None:
        raise ValueError('X is equal to None!')

    if extend and len(X.shape) == 2:
        warnings.warn('2-Dimensional data passed. Assuming these are '
                      '{} 1-dimensional timeseries'.format(X.shape[0]))
        X = X.reshape((X.shape) + (1,))

    if X_fit_dims is not None:
        if check_n_features_only:
            if X_fit_dims[2] != X.shape[2]:
                raise ValueError(
                    'Number of features of the provided timeseries'
                    '(last dimension) must match the one of the fitted data!'
                    ' ({} and {} are passed shapes)'.format(X_fit_dims,
                                                            X.shape))
        else:
            if X_fit_dims[1:] != X.shape[1:]:
                raise ValueError(
                    'Dimensions of the provided timeseries'
                    '(except first) must match those of the fitted data!'
                    ' ({} and {} are passed shapes)'.format(X_fit_dims,
                                                            X.shape))

    return X


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
        The transformed time series. This is always guaraneteed to be a new
        time series and never just a view into the old one.

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
    ts_out = numpy.array(ts, copy=True)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != float:
        ts_out = ts_out.astype(float)
    if remove_nans:
        ts_out = ts_out[:ts_size(ts_out)]
    return ts_out


def to_time_series_dataset(dataset, dtype=float):
    """Transforms a time series dataset so that it fits the format used in
    ``tslearn`` models.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed. A single time series will
        be automatically wrapped into a dataset with a single entry.
    dtype : data type (default: float)
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
    >>> to_time_series_dataset([1, 2])
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
    >>> to_time_series_dataset([]).shape
    (0, 0, 0)

    See Also
    --------
    to_time_series : Transforms a single time series
    """
    try:
        import pandas as pd
        if isinstance(dataset, pd.DataFrame):
            return to_time_series_dataset(numpy.array(dataset))
    except ImportError:
        pass
    if isinstance(dataset, NotAnArray):  # Patch to pass sklearn tests
        return to_time_series_dataset(numpy.array(dataset))
    if len(dataset) == 0:
        return numpy.zeros((0, 0, 0))
    if numpy.array(dataset[0]).ndim == 0:
        dataset = [dataset]
    n_ts = len(dataset)
    max_sz = max([ts_size(to_time_series(ts, remove_nans=True))
                  for ts in dataset])
    d = to_time_series(dataset[0]).shape[1]
    dataset_out = numpy.zeros((n_ts, max_sz, d), dtype=dtype) + numpy.nan
    for i in range(n_ts):
        ts = to_time_series(dataset[i], remove_nans=True)
        dataset_out[i, :ts.shape[0]] = ts
    return dataset_out.astype(dtype)


def time_series_to_str(ts, fmt="%.18e"):
    """Transforms a time series to its representation as a string (used when
    saving time series to disk).

    Parameters
    ----------
    ts : array-like
        Time series to be represented.
    fmt : string (default: "%.18e")
        Format to be used to write each value (only ASCII characters).

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
    out = StringIO()
    numpy.savetxt(out, ts_.T, fmt=fmt, delimiter=" ", newline="|", encoding="bytes")
    return out.getvalue()[:-1]  # cut away the trailing "|"


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
    ts = [numpy.fromstring(dim_str, sep=" ") for dim_str in dimensions]
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
    with open(fname, "w") as f:
        for ts in dataset:
            f.write(time_series_to_str(ts, fmt=fmt) + "\n")


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
    with open(fname, "r") as f:
        return to_time_series_dataset([
            str_to_time_series(row)
            for row in f.readlines()
        ])


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
    >>> check_equal_size([])
    True
    """
    dataset_ = to_time_series_dataset(dataset)
    if len(dataset_) == 0:
        return True

    size = ts_size(dataset[0])
    return all(ts_size(ds) == size for ds in dataset_[1:])


def ts_size(ts):
    """Returns actual time series size.

    Final timesteps that have `NaN` values for all dimensions will be removed
    from the count. Infinity and negative infinity ar considered valid time
    series values.

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
    >>> ts_size([numpy.nan, 3, numpy.inf, numpy.nan])
    3
    """
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    while sz > 0 and numpy.all(numpy.isnan(ts_[sz - 1])):
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


class LabelCategorizer(TransformerMixin, TimeSeriesBaseEstimator):
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
        out = TimeSeriesBaseEstimator.get_params(self, deep=deep)
        out["single_column_if_binary"] = self.single_column_if_binary
        out["forward_match"] = self.forward_match
        out["backward_match"] = self.backward_match
        return out

    def _more_tags(self):
        return {'X_types': ['1dlabels']}


def _load_arff_uea(dataset_path):
    """Load arff file for uni/multi variate dataset
    
    Parameters
    ----------
    dataset_path: string of dataset_path
        Path to the ARFF file to be read

    Returns
    -------
    x: numpy array of shape (n_timeseries, n_timestamps, n_features)
        Time series dataset
    y: numpy array of shape (n_timeseries, )
        Vector of targets

    Raises
    ------
    ImportError: if the version of *Scipy* is too old (pre 1.3.0)
    Exception: on any failure, e.g. if the given file does not exist or is
               corrupted
    """
    if not HAS_ARFF:
        raise ImportError("scipy 1.3.0 or newer is required to load "
                          "time series datasets from arff format.")
    data, meta = arff.loadarff(dataset_path)
    names = meta.names()  # ["input", "class"] for multi-variate

    # firstly get y_train
    y_ = data[names[-1]]  # data["class"]
    y = numpy.array(y_).astype("str")

    # get x_train
    if len(names) == 2:  # len=2 => multi-variate
        x_ = data[names[0]]
        x_ = numpy.asarray(x_.tolist())

        nb_example = x_.shape[0]
        nb_channel = x_.shape[1]
        length_one_channel = len(x_.dtype.descr)
        x = numpy.empty([nb_example, length_one_channel, nb_channel])

        for i in range(length_one_channel):
            # x_.dtype.descr: [('t1', '<f8'), ('t2', '<f8'), ('t3', '<f8')]
            time_stamp = x_.dtype.descr[i][0]  # ["t1", "t2", "t3"]
            x[:, i, :] = x_[time_stamp]

    else:  # uni-variate situation
        x_ = data[names[:-1]]
        x = numpy.asarray(x_.tolist(), dtype=float)
        x = x.reshape(len(x), -1, 1)

    return x, y


def _load_txt_uea(dataset_path):
    """Load arff file for uni/multi variate dataset

    Parameters
    ----------
    dataset_path: string of dataset_path
        Path to the TXT file to be read

    Returns
    -------
    x: numpy array of shape (n_timeseries, n_timestamps, n_features)
        Time series dataset
    y: numpy array of shape (n_timeseries, )
        Vector of targets

    Raises
    ------
    Exception: on any failure, e.g. if the given file does not exist or is
               corrupted
    """
    data = numpy.loadtxt(dataset_path)
    X = to_time_series_dataset(data[:, 1:])
    y = data[:, 0].astype(int)
    return X, y
