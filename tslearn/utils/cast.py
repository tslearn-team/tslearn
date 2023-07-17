import warnings

import numpy
from sklearn.utils import check_array

try:
    from scipy.io import arff
    HAS_ARFF = True
except:
    HAS_ARFF = False

from .utils import check_dataset, ts_size, to_time_series_dataset


def to_sklearn_dataset(dataset, dtype=float, return_dim=False):
    """Transforms a time series dataset so that it fits the format used in
    ``sklearn`` estimators.

    Parameters
    ----------
    dataset : array-like
        The dataset of time series to be transformed.
    dtype : data type (default: float64)
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
    return numpy.array([Xi[:ts_size(Xi)] for Xi in X_], dtype=object)


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
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Conversion from/to sktime cannot be performed "
                          "if pandas is not installed.")
    X_ = check_dataset(X)
    X_pd = pd.DataFrame(dtype=float)
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
    >>> import pandas as pd
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
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Conversion from/to sktime cannot be performed "
                          "if pandas is not installed.")
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
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Conversion from/to pyflux cannot be performed "
                          "if pandas is not installed.")
    X_ = check_dataset(X,
                       force_equal_length=True,
                       force_single_time_series=True)
    X_pd = pd.DataFrame(X[0], dtype=float)
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
    >>> import pandas as pd
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
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Conversion from/to pyflux cannot be performed "
                          "if pandas is not installed.")
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
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Conversion from/to tsfresh cannot be performed "
                          "if pandas is not installed.")
    X_ = check_dataset(X)
    n, sz, d = X_.shape
    dataframes = []
    for i, Xi in enumerate(X_):
        df = pd.DataFrame(columns=["id", "time"] +
                                  ["dim_%d" % di for di in range(d)])
        Xi_ = Xi[:ts_size(Xi)]
        sz = Xi_.shape[0]
        df["time"] = numpy.arange(sz)
        df["id"] = numpy.zeros((sz,), dtype=int) + i
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
    >>> import pandas as pd
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
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Conversion from/to tsfresh cannot be performed "
                          "if pandas is not installed.")
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
    try:
        from cesium.time_series import TimeSeries
    except ImportError:
        raise ImportError("Conversion from/to cesium cannot be performed "
                          "if cesium is not installed.")

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
    >>> from cesium.time_series import TimeSeries
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
    try:
        from cesium.time_series import TimeSeries
    except ImportError:
        raise ImportError("Conversion from/to cesium cannot be performed "
                          "if cesium is not installed.")

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
