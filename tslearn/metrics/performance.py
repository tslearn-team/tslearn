"""
The :mod:`tslearn.metrics.performance` module delivers time-series specific performance metrics .

"""

from sklearn import metrics as skmetrics

from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series_dataset


def mae(
    y_true,
    y_pred,
    ts_weights=None,
    timestamps_weights=None,
    multioutput='uniform_average'
):
    """
    Mean absolute error (MAE)

    MAE is a measure of the prediction accuracy for forecasting and regression tasks that computes
    the average of the deviations between ground truth and predicted values.

    Parameters
    ----------
    y_true: array like, shape (n_ts, sz, d)
        Target dataset of ground_truth values
    y_pred: array like, shape (n_ts, sz, d)
        Estimated dataset of predicted values
    ts_weights: array like, shape (n_ts,) or None (default: None)
        Weights to apply to the time-series in the datasets if non-uniform.
        Use none for uniform weights.
    timestamps_weights: array like, shape (sz,) or None (default: None)
        Weights to apply to the timestamps of each-time series if non-uniform.
        Use none for uniform weights.
    multioutput: {'uniform_average', 'raw_values'} or array-like, shape (d,) (default: 'uniform_average')
        for multivariate timeseries, defines the aggregation of per feature results, if any.

        'raw_values':
            no aggregation, result is per feature
        'uniform_average':
            errors of all features are averaged with uniform weights
        array-like:
            errors of all features are averaged using the given weights

    Returns
    -------
    float or array-like, shape (d,)
        If multioutput is ‘raw_values’, then mean absolute error is returned for each feature.
        Otherwise, the average of each feature is returned.

    """

    y_true = to_time_series_dataset(y_true)
    y_pred = to_time_series_dataset(y_pred)

    n_ts, sz, d = y_true.shape

    res = skmetrics.mean_absolute_error(
        y_true.reshape(n_ts, sz * d),
        y_pred.reshape(n_ts, sz * d),
        sample_weight=ts_weights,
        multioutput="raw_values"
    ).reshape(sz, d)
    be = instantiate_backend(res)

    if timestamps_weights is not None:
        timestamps_weights = be.asarray(timestamps_weights)
        res = timestamps_weights @ res / timestamps_weights.sum()
    else:
        res = res.mean(axis=0)

    if isinstance(multioutput, str):
        if multioutput == "uniform_average":
            res = res.mean()
    else:
        multioutput = be.asarray(multioutput)
        res = multioutput@res / multioutput.sum()
    return res


def mse(
    y_true,
    y_pred,
    ts_weights=None,
    timestamps_weights=None,
    multioutput="uniform_average"
):
    """
    Mean squared error (MSE)

    MSE is a measure of the prediction accuracy for forecasting and regression tasks that computes
    the average of the squared deviations between ground truth and predicted values.

    Parameters
    ----------
    y_true: array like, shape (n_ts, sz, d)
        Target dataset of ground_truth values
    y_pred: array like, shape (n_ts, sz, d)
        Estimated dataset of predicted values
    ts_weights: array like, shape (n_ts,) or None (default: None)
        Weights to apply to the time-series in the datasets if non-uniform.
        Use none for uniform weights.
    timestamps_weights: array like, shape (sz,) or None (default: None)
        Weights to apply to the timestamps of each-time series if non-uniform.
        Use none for uniform weights.
    multioutput: {'uniform_average', 'raw_values'} or array-like, shape (d,) (default: 'uniform_average')
        for multivariate timeseries, defines the aggregation of per feature results, if any.

        'raw_values':
            no aggregation, result is per feature
        'uniform_average':
            errors of all features are averaged with uniform weights
        array-like:
            errors of all features are averaged using the given weights

    Returns
    -------
    float or array-like, shape (d,)
        If multioutput is ‘raw_values’, then mean squared error is returned for each feature.
        Otherwise, the average of each feature is returned.

    """
    y_true = to_time_series_dataset(y_true)
    y_pred = to_time_series_dataset(y_pred)

    n_ts, sz, d = y_true.shape

    res = skmetrics.mean_squared_error(
        y_true.reshape(n_ts, sz * d),
        y_pred.reshape(n_ts, sz * d),
        sample_weight=ts_weights,
        multioutput="raw_values"
    ).reshape(sz, d)
    be = instantiate_backend(res)

    if timestamps_weights is not None:
        timestamps_weights = be.asarray(timestamps_weights)
        res = timestamps_weights @ res / timestamps_weights.sum()
    else:
        res = res.mean(axis=0)

    if isinstance(multioutput, str):
        if multioutput == "uniform_average":
            res = res.mean()
    else:
        multioutput = be.asarray(multioutput)
        res = multioutput@res / multioutput.sum()
    return res


def mase(
    y_true,
    y_pred,
    train_data,
    seasonal_period=1,
    ts_weights=None,
    timestamps_weights=None,
    multioutput="uniform_average"
):
    """
    Mean absolute scaled error (MASE)

    MASE is a measure of the prediction accuracy for forecasting and regression tasks that computes
    the scaled average of the deviations between ground truth and predicted values.

    The scaling factor is computed as the MAE of the naive m-seasonal forecast on the in-sample dataset.

    Parameters
    ----------
    y_true: array like, shape (n_ts, sz, d)
        Target dataset of ground_truth values
    y_pred: array like, shape (n_ts, sz, d)
        Estimated dataset of predicted values
    train_data: array like
        the in-sample dataset, used to compute the scaling factor
    seasonal_period: int (default: 1)
        seasonal period used to compute the scaling factor
    ts_weights: array like, shape (n_ts,) or None (default: None)
        Weights to apply to the time-series in the datasets if non-uniform.
        Use none for uniform weights.
    timestamps_weights: array like, shape (sz,) or None (default: None)
        Weights to apply to the timestamps of each-time series if non-uniform.
        Use none for uniform weights.
    multioutput: {'uniform_average', 'raw_values'} or array-like, shape (d,) (default: 'uniform_average')
        for multivariate timeseries, defines the aggregation of per feature results, if any.

        'raw_values':
            no aggregation, result is per feature
        'uniform_average':
            errors of all features are averaged with uniform weights
        array-like:
            errors of all features are averaged using the given weights

    Returns
    -------
    float or array-like, shape (d,)
        If multioutput is ‘raw_values’, then mean squared error is returned for each feature.
        Otherwise, the average of each feature is returned.

    """

    train_data = to_time_series_dataset(train_data)

    mae_ = mae(y_true, y_pred, ts_weights, timestamps_weights, multioutput)

    if multioutput == "raw_values":
        scale_axis = (0, 1)
    else:
        scale_axis = None
    scale = abs(train_data[:, :-seasonal_period] - train_data[:, seasonal_period:]).mean(axis=scale_axis)

    return mae_ / scale
