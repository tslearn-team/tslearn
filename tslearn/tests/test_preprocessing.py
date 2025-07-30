import math

import numpy as np

import pytest

from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
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


def test_scaler_allow_variable_length():
    variable_length_dataset = [[1, 2], [1, 2, 3]]

    for estimator_cls in [TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax]:
        estimator = estimator_cls()
        tags = estimator._get_tags()

        assert ALLOW_VARIABLE_LENGTH in tags
        assert not tags[ALLOW_VARIABLE_LENGTH]

        with pytest.raises(ValueError):
            estimator.fit_transform(variable_length_dataset)


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
        [[1, 2], [2, 3], [2, math.nan]],
        [[3, 4], [math.nan, 5]],
    ]
    univariate_dataset = [
        [1, math.nan, 3],
        [1, 2, math.nan, 9]
    ]

    # Test default params and equivalent mean method
    imputer = TimeSeriesImputer()
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, 2], [2, 3], [2, 2.5]],
        [[3, 4], [3, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = TimeSeriesImputer(method="mean").fit_transform(multivariate_dataset)
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [2], [3], [np.nan]],
        [[1], [2], [4], [9]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = TimeSeriesImputer(method="mean").fit_transform(univariate_dataset)
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
        [[1, math.nan], [2, 3], [2, math.nan]],
        [[3, 4], [math.nan, 5]],
    ]
    univariate_dataset = [
        [1, math.nan, math.nan, 3, 6, math.nan, 9],
        [1, 2, math.nan, 9],
        [math.nan, 2, math.nan, 9, 6, math.nan],
    ]
    imputer.set_params(method="ffill")
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, math.nan], [2, 3], [2, 3]],
        [[3, 4], [3, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [1], [1], [3], [6], [6], [9] ],
        [[1], [2], [2], [9], [math.nan], [math.nan], [math.nan] ],
        [[math.nan], [2], [2], [9], [6], [6], [math.nan]],
    ])
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
        [[2], [2], [9], [9], [6], [math.nan], [math.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)

    # Constant:
    # with default value: no changes except for NaN padding
    # with value : non padded nans filled with value
    imputer.set_params(method="constant")
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, math.nan], [2, 3], [2, math.nan]],
        [[3, 4], [math.nan, 5], [math.nan, math.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [math.nan], [math.nan], [3], [6], [math.nan], [9]],
        [[1], [2], [math.nan], [9], [math.nan], [math.nan], [math.nan]],
        [[math.nan], [2], [math.nan], [9], [6], [math.nan], [math.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    value=42.42
    imputer.set_params(value=value)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, value], [2, 3], [2, value]],
        [[3, 4], [value, 5], [math.nan, math.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [value], [value], [3], [6], [value], [9]],
        [[1], [2], [value], [9], [math.nan], [math.nan], [math.nan]],
        [[value], [2], [value], [9], [6], [value], [math.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)

    multivariate_dataset = [
        [[1, math.nan], [2, 3], [2, math.nan]],
        [[3, 4], [math.nan, 5], [math.nan, math.nan]],
    ]
    univariate_dataset = [
        [1, math.nan, math.nan, 3, 6, math.nan, 9],
        [1, 2, math.nan, 9],
        [math.nan, 2, math.nan, 9, 6, math.nan],
    ]
    imputer.set_params(keep_trailing_nans=True)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, value], [2, 3], [2, value]],
        [[3, 4], [value, 5], [math.nan, math.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [value], [value], [3], [6], [value], [9]],
        [[1], [2], [value], [9], [math.nan], [math.nan], [math.nan]],
        [[value], [2], [value], [9], [6], [math.nan], [math.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)

    imputer.set_params(method=lambda x: to_time_series([1, 2, 3]))
    transformed = imputer.fit_transform([[1, math.nan, 3]])
    expected = np.array([
        [[1.], [2.], [3.]]
    ])
    np.testing.assert_array_equal(transformed, expected)

    imputer.set_params(method="unknown")
    with pytest.raises(ValueError):
        imputer.fit_transform([[1, math.nan, 3]])
