import numpy as np

from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.preprocessing import (TimeSeriesScalerMeanVariance,
                                   TimeSeriesScalerMinMax)
from tslearn.tests.sklearn_patches import assert_raises
from tslearn.utils import to_time_series_dataset


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

        with assert_raises(ValueError):
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
