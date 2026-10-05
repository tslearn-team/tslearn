import numpy as np

import pytest

from tslearn.generators import random_walks
from tslearn.preprocessing import (TimeSeriesScalerMeanVariance,
                                   TimeSeriesScalerMinMax,
                                   TimeSeriesImputer)
from tslearn.utils import to_time_series_dataset, to_time_series


def test_single_value_ts_no_nan():
    X = to_time_series_dataset([[1, 1, 1, 1]])

    standard_scaler = TimeSeriesScalerMeanVariance()
    assert np.sum(np.isnan(standard_scaler.fit_transform(X))) == 0

    minmax_scaler = TimeSeriesScalerMinMax()
    assert np.sum(np.isnan(minmax_scaler.fit_transform(X))) == 0


def test_min_max_scaler_range():
    with pytest.raises(ValueError):
        TimeSeriesScalerMinMax((1., 0.)).fit_transform([[1, 2, 3]])


def test_min_max_scaler_variable_length():
    X = [
        [1, np.nan],
        [3, 4]
    ]

    estimator = TimeSeriesScalerMinMax(per_timeseries=True)
    transformed = estimator.fit_transform(X)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0], [np.nan]],
            [[0], [1]],
        ])
    )
    transformed = estimator.transform([[1, 2, 3]])
    np.testing.assert_array_equal(
        transformed,
        np.array([[[0], [0.5], [1]]])
    )


@pytest.mark.parametrize(
    "scaler", [TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance]
)
@pytest.mark.parametrize("per_timeseries", [True, False])
@pytest.mark.parametrize("per_feature", [True, False])
@pytest.mark.parametrize("n_features", [1, 2])
def test_scaler_ragged_input(scaler, per_timeseries, per_feature, n_features):
    X = [np.array([[1., 10.], [3., 20.]])[:, :n_features],
         np.array([[5., 30.], [7., 40.], [9., 50.]])[:, :n_features]]
    query = [np.array([[2., 15.]])[:, :n_features],
             np.array([[4., 25.], [6., 35.], [8., 45.]])[:, :n_features]]
    X_padded = to_time_series_dataset(X)
    query_padded = to_time_series_dataset(query)
    params = dict(per_timeseries=per_timeseries, per_feature=per_feature)
    reference = scaler(**params)
    expected = reference.fit_transform(X_padded)

    estimator = scaler(**params)
    np.testing.assert_allclose(estimator.fit_transform(X), expected)
    np.testing.assert_allclose(
        estimator.transform(query), reference.transform(query_padded)
    )
    for original, ts in zip(X, estimator.transform(X)):
        np.testing.assert_array_equal(np.isnan(ts[len(original):]), True)


@pytest.mark.parametrize(
    "scaler", [TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance]
)
@pytest.mark.parametrize("per_feature", [True, False])
@pytest.mark.parametrize("n_features", [1, 2])
def test_scaler_inverse_ragged_input(scaler, per_feature, n_features):
    X = [np.array([[1., 10.], [3., 20.]])[:, :n_features],
         np.array([[5., 30.], [7., 40.], [9., 50.]])[:, :n_features]]
    X_padded = to_time_series_dataset(X)
    estimator = scaler(per_timeseries=False, per_feature=per_feature)
    transformed = estimator.fit_transform(X_padded)
    ragged = [ts[:len(original)] for original, ts in zip(X, transformed)]
    np.testing.assert_allclose(estimator.inverse_transform(ragged), X_padded)


@pytest.mark.parametrize(
    "scaler", [TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance]
)
@pytest.mark.parametrize("method", ["fit", "transform", "inverse_transform"])
def test_scaler_empty_list(scaler, method):
    estimator = scaler(per_timeseries=False).fit([[1., 3.]])
    with pytest.raises(ValueError):
        getattr(estimator, method)([])


@pytest.mark.parametrize(
    "scaler", [TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance]
)
@pytest.mark.parametrize("method, per_timeseries", [
    ("fit", True), ("fit", False), ("transform", True),
    ("transform", False), ("inverse_transform", False)
])
@pytest.mark.parametrize("per_feature", [True, False])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("lengths", [(2, 2), (2, 3)])
def test_scaler_mixed_features(scaler, method, per_timeseries, per_feature,
                               reverse, lengths):
    X = [np.arange(lengths[0] * 2).reshape(-1, 2),
         np.arange(lengths[1]).reshape(-1, 1)]
    if reverse:
        X.reverse()
    estimator = scaler(per_timeseries=per_timeseries,
                       per_feature=per_feature).fit([[[1., 10.], [3., 20.]]])
    with pytest.raises(ValueError, match="same number of features"):
        getattr(estimator, method)(X)


@pytest.mark.parametrize(
    "scaler", [TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance]
)
@pytest.mark.parametrize("method", ["fit", "transform", "inverse_transform"])
@pytest.mark.parametrize("reverse", [True, False])
def test_scaler_empty_series(scaler, method, reverse):
    X = [np.empty((0, 2)), np.array([[1., 10.], [3., 20.]])]
    if reverse:
        X.reverse()
    estimator = scaler(per_timeseries=False).fit([[[1., 10.], [3., 20.]]])
    with pytest.raises(ValueError):
        getattr(estimator, method)(X)


@pytest.mark.parametrize(
    "scaler", [TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance]
)
@pytest.mark.parametrize("method, per_timeseries", [
    ("fit", True), ("fit", False), ("transform", True),
    ("transform", False), ("inverse_transform", False)
])
@pytest.mark.parametrize("per_feature", [True, False])
def test_scaler_array_protocol(scaler, method, per_timeseries, per_feature):
    class ArrayProtocolDataset:
        def __array__(self, dtype=None, copy=None):
            array = np.asarray(X, dtype=dtype)
            return array.copy() if copy else array

    X = np.array([[[1., 10.], [3., 20.]], [[5., 30.], [7., 40.]]])
    params = dict(per_timeseries=per_timeseries, per_feature=per_feature)
    estimator = scaler(**params).fit(X)
    reference = scaler(**params).fit(X)
    if method == "fit":
        np.testing.assert_allclose(
            estimator.fit(ArrayProtocolDataset()).transform(X),
            reference.transform(X)
        )
    else:
        np.testing.assert_allclose(
            getattr(estimator, method)(ArrayProtocolDataset()),
            getattr(reference, method)(X)
        )


@pytest.mark.parametrize(
    "scaler", [TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance]
)
@pytest.mark.parametrize("method", ["fit", "transform", "inverse_transform"])
@pytest.mark.parametrize("X", [None, 1, 1., np.float64(1), np.array(1)])
def test_scaler_scalar_input(scaler, method, X):
    estimator = scaler(per_timeseries=False).fit([[1., 3.]])
    with pytest.raises(ValueError):
        getattr(estimator, method)(X)


@pytest.mark.parametrize(
    "scaler",
    [TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance]
)
@pytest.mark.parametrize(
    "per_timeseries, per_feature",
    [(True, True), (True, False), (False, True), (False, False)]
)
def test_scaler_inverse_transform(scaler, per_timeseries, per_feature):
    X = random_walks(10, 10, 2, mu=1, random_state=0)

    estimator = scaler(per_timeseries=per_timeseries, per_feature=per_feature)
    transformed = estimator.fit_transform(X)
    if per_timeseries:
        with pytest.raises(RuntimeError):
            estimator.inverse_transform(X)
    else:
        np.testing.assert_array_almost_equal(
        estimator.inverse_transform(transformed),
        X
    )


def test_mean_variance_inverse_nonzero_mu():
    X = np.array([[[0.], [2.]]])
    estimator = TimeSeriesScalerMeanVariance(mu=5.0, std=1.0, per_timeseries=False)
    transformed = estimator.fit_transform(X)
    np.testing.assert_array_almost_equal(transformed, np.array([[[4.], [6.]]]))
    np.testing.assert_array_almost_equal(estimator.inverse_transform(transformed), X)


def test_min_max_scaler_constant_inverse():
    X = np.array([[[3., 1.], [3., 2.]], [[3., 3.], [3., 4.]]])
    query = np.array([[[4., 2.], [1., 5.]]])
    estimator = TimeSeriesScalerMinMax(per_timeseries=False).fit(X)
    np.testing.assert_array_almost_equal(
        estimator.inverse_transform(estimator.transform(query)),
        query
    )


def test_min_max_scaler_modes():
    univariate_dataset = [
        [1, 2, 3],
        [3, 4, 5]
    ]
    multivariate_dataset = [
        [[1, 2], [2, 3]],
        [[3, 4], [4, 5]],
    ]

    estimator_cls = TimeSeriesScalerMinMax
    estimator = estimator_cls(per_feature=True, per_timeseries=True)
    transformed = estimator.fit_transform(multivariate_dataset)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ])
    )
    transformed = estimator.fit_transform(univariate_dataset)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0], [0.5], [1]],
            [[0], [0.5], [1]],
        ])
    )

    estimator = estimator_cls(per_feature=False, per_timeseries=True)
    transformed = estimator.fit_transform(multivariate_dataset)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0, 0.5], [0.5, 1]],
            [[0, 0.5], [0.5, 1]],
        ])
    )
    transformed = estimator.fit_transform(univariate_dataset)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0], [0.5], [1]],
            [[0], [0.5], [1]],
        ])
    )

    estimator = estimator_cls(per_feature=True, per_timeseries=False)
    transformed = estimator.fit_transform(multivariate_dataset)
    np.testing.assert_array_almost_equal(
        transformed,
        np.array([
            [[0, 0], [0.33, 0.33]],
            [[0.66, 0.66], [1, 1]],
        ]),
        decimal=2
    )
    transformed = estimator.fit_transform(univariate_dataset)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0], [0.25], [0.5]],
            [[0.5], [0.75], [1]],
        ])
    )

    estimator = estimator_cls(per_feature=False, per_timeseries=False)
    transformed = estimator.fit_transform(multivariate_dataset)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0, 0.25], [0.25, 0.5]],
            [[0.5, 0.75], [0.75, 1]],
        ])
    )
    transformed = estimator.fit_transform(univariate_dataset)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0], [0.25], [0.5]],
            [[0.5], [0.75], [1]],
        ])
    )


def test_mean_variance_scaler_variable_length():
    X = [
        [1, np.nan],
        [3, 4]
    ]

    estimator = TimeSeriesScalerMeanVariance(per_timeseries=True)
    transformed = estimator.fit_transform(X)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[0], [np.nan]],
            [[-1], [1]],
        ])
    )
    transformed = estimator.transform([[1, 2, 3]])
    np.testing.assert_array_equal(
        transformed,
        np.array([[[-np.sqrt(3/2)], [0], [np.sqrt(3/2)]]])
    )


def test_mean_variance_scaler_modes():
    univariate_dataset = [
        [1, 2, 3],
        [3, 4, 5]
    ]
    multivariate_dataset = [
        [[1, 2], [2, 3]],
        [[3, 4], [4, 5]],
    ]

    estimator_cls = TimeSeriesScalerMeanVariance
    estimator = estimator_cls(per_feature=True, per_timeseries=True)
    transformed = estimator.fit_transform(multivariate_dataset)
    np.testing.assert_array_equal(
        transformed,
        np.array([
            [[-1, -1], [1, 1]],
            [[-1, -1], [1, 1]],
        ])
    )
    transformed = estimator.fit_transform(univariate_dataset)
    np.testing.assert_array_almost_equal(
        transformed,
        np.array([
            [[-1.22], [0], [1.22]],
            [[-1.22], [0], [1.22]],
        ]),
        decimal=2
    )

    estimator = estimator_cls(per_feature=False, per_timeseries=True)
    transformed = estimator.fit_transform(multivariate_dataset)
    np.testing.assert_array_almost_equal(
        transformed,
        np.array([
            [[-1.41, 0], [0, 1.41]],
            [[-1.41, 0], [0, 1.41]],
        ]),
        decimal=2
    )
    transformed = estimator.fit_transform(univariate_dataset)
    np.testing.assert_array_almost_equal(
        transformed,
        np.array([
            [[-1.22], [0], [1.22]],
            [[-1.22], [0], [1.22]],
        ]),
        decimal=2
    )

    estimator = estimator_cls(per_feature=True, per_timeseries=False)
    transformed = estimator.fit_transform(multivariate_dataset)
    np.testing.assert_array_almost_equal(
        transformed,
        np.array([
            [[-1.34, -1.34], [-0.44, -0.44]],
            [[0.44, 0.44], [1.34, 1.34]],
        ]),
        decimal=2
    )
    transformed = estimator.fit_transform(univariate_dataset)
    np.testing.assert_array_almost_equal(
        transformed,
        np.array([
            [[-1.54], [-0.77], [0]],
            [[0], [0.77], [1.54]],
        ]),
        decimal=2
    )

    estimator = estimator_cls(per_feature=False, per_timeseries=False)
    transformed = estimator.fit_transform(multivariate_dataset)
    np.testing.assert_array_almost_equal(
        transformed,
        np.array([
            [[-1.63, -0.81], [-0.81, 0]],
            [[0, 0.81], [0.81, 1.63]],
        ]),
        decimal=2
    )
    transformed = estimator.fit_transform(univariate_dataset)
    np.testing.assert_array_almost_equal(
        transformed,
        np.array([
            [[-1.54], [-0.77], [0]],
            [[0], [0.77], [1.54]],
        ]),
        decimal=2
    )


def test_imputer():
    multivariate_dataset = [
        [[1, 2], [2, 3], [2, np.nan]],
        [[3, 4], [np.nan, 5]],
    ]
    univariate_dataset = [
        [1, np.nan, 3],
        [1, 2, np.nan, 9]
    ]

    # Default method is mean
    imputer = TimeSeriesImputer(keep_trailing_nans=True)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, 2], [2, 3], [2, 2.5]],
        [[3, 4], [3, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = TimeSeriesImputer(
        method="mean",
        keep_trailing_nans=True
    ).fit_transform(multivariate_dataset)
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [2], [3], [np.nan]],
        [[1], [2], [4], [9]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = TimeSeriesImputer(
        method="mean",
        keep_trailing_nans=True
    ).fit_transform(univariate_dataset)
    np.testing.assert_array_equal(transformed, expected)

    # Test median method
    imputer.set_params(method="median")
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, 2], [2, 3], [2, 2.5]],
        [[3, 4], [3, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [2], [3], [np.nan]],
        [[1], [2], [2], [9]],
    ])
    np.testing.assert_array_equal(transformed, expected)

    # Test ffill method
    multivariate_dataset = [
        [[1, np.nan], [2, 3], [2, np.nan]],
        [[3, 4], [np.nan, 5]],
    ]
    univariate_dataset = [
        [1, np.nan, np.nan, 3, 6, np.nan, 9],
        [1, 2, np.nan, 9],
        [np.nan, 2, np.nan, 9, 6, np.nan],
    ]
    imputer.set_params(method="ffill")
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, np.nan], [2, 3], [2, 3]],
        [[3, 4], [3, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array(
        [
            [[1], [1], [1], [3], [6], [6], [9]],
            [[1], [2], [2], [9], [np.nan], [np.nan], [np.nan]],
            [[np.nan], [2], [2], [9], [6], [np.nan], [np.nan]],
        ]
    )
    np.testing.assert_array_equal(transformed, expected)

    # Test bfill method
    imputer.set_params(method="bfill")
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, 3], [2, 3], [2, np.nan]],
        [[3, 4], [np.nan, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [3], [3], [3], [6], [9], [9]],
        [[1], [2], [9], [9], [np.nan], [np.nan], [np.nan]],
        [[2], [2], [9], [9], [6], [np.nan], [np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)

    # Constant:
    # with default value: no changes except for NaN padding
    # with value : non padded nans filled with value
    imputer.set_params(method="constant")
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, np.nan], [2, 3], [2, np.nan]],
        [[3, 4], [np.nan, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [np.nan], [np.nan], [3], [6], [np.nan], [9]],
        [[1], [2], [np.nan], [9], [np.nan], [np.nan], [np.nan]],
        [[np.nan], [2], [np.nan], [9], [6], [np.nan], [np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    value = 42.42
    imputer.set_params(value=value)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, value], [2, 3], [2, value]],
        [[3, 4], [value, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array(
        [
            [[1], [value], [value], [3], [6], [value], [9]],
            [[1], [2], [value], [9], [np.nan], [np.nan], [np.nan]],
            [[value], [2], [value], [9], [6], [np.nan], [np.nan]],
        ]
    )
    np.testing.assert_array_equal(transformed, expected)

    imputer.set_params(method="linear")
    multivariate_dataset = [
        [[1, np.nan], [np.nan, 3], [2, np.nan]],
        [[3, 4], [np.nan, 5], [6, np.nan], [np.nan, 7]],
    ]
    univariate_dataset = [
        [1, np.nan, np.nan, 3, 6, np.nan, 9],
        [1, 2, np.nan, 9],
        [np.nan, 2, np.nan, 9, 6, np.nan],
    ]
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array(
        [
            [[1], [5 / 3], [7 / 3], [3], [6], [7.5], [9]],
            [[1], [2], [5.5], [9], [np.nan], [np.nan], [np.nan]],
            [[2], [2], [5.5], [9], [6], [np.nan], [np.nan]],
        ]
    )
    np.testing.assert_array_almost_equal(transformed, expected)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, 3], [1.5, 3], [2, 3], [np.nan, np.nan]],
        [[3, 4], [4.5, 5], [6, 6], [6, 7]],
    ])
    np.testing.assert_array_almost_equal(transformed, expected)

    # A feature with no observed value is left unchanged
    transformed = imputer.fit_transform(
        [[[1, np.nan], [np.nan, np.nan], [3, np.nan]]]
    )
    expected = np.array([[[1, np.nan], [2, np.nan], [3, np.nan]]])
    np.testing.assert_array_equal(transformed, expected)

    multivariate_dataset = [
        [[1, np.nan], [2, 3], [2, np.nan]],
        [[3, 4], [np.nan, 5], [np.nan, np.nan]],
    ]
    univariate_dataset = [
        [1, np.nan, np.nan, 3, 6, np.nan, 9],
        [1, 2, np.nan, 9],
        [np.nan, 2, np.nan, 9, 6, np.nan],
    ]
    imputer.set_params(method="constant", keep_trailing_nans=False)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array(
        [
            [[1, value], [2, 3], [2, value]],
            [[3, 4], [value, 5], [value, value]],
        ]
    )
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array(
        [
            [[1], [value], [value], [3], [6], [value], [9]],
            [[1], [2], [value], [9], [value], [value], [value]],
            [[value], [2], [value], [9], [6], [value], [value]],
        ]
    )
    np.testing.assert_array_equal(transformed, expected)

    transformed = imputer.fit_transform([[1, np.nan]])
    expected = np.array([[[1.0]]])
    np.testing.assert_array_equal(transformed, expected)
    imputer.set_params(keep_trailing_nans=True)
    transformed = imputer.fit_transform([[1, np.nan]])
    np.testing.assert_array_equal(transformed, expected)

    imputer.set_params(method=lambda x: to_time_series([1, 2, 3]))
    transformed = imputer.fit_transform([[1, np.nan, 3]])
    expected = np.array([
        [[1.], [2.], [3.]]
    ])
    np.testing.assert_array_equal(transformed, expected)

    imputer.set_params(method="unknown")
    with pytest.raises(ValueError):
        imputer.fit_transform([[1, np.nan, 3]])
