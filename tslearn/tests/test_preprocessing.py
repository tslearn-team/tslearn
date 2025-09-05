import numpy as np
from numpy.testing import assert_array_equal

import pytest

from tslearn.bases.bases import ALLOW_VARIABLE_LENGTH
from tslearn.preprocessing import (
    TimeSeriesScalerMeanVariance,
    TimeSeriesScalerMinMax,
    TimeSeriesImputer,
    TimeSeriesResampler
)

from tslearn.utils import to_time_series_dataset, to_time_series


def test_resampler_invalid_method():
    with pytest.raises(ValueError):
        TimeSeriesResampler(method="invalid").fit_transform([[1, 2, 3]])


def test_resampler_uniform():

    # Test downsampling
    X = to_time_series_dataset([1, 2, 3, 4, 5, 6, 7])
    resampled = TimeSeriesResampler(sz=3, method="uniform").fit_transform(X)
    expected = np.array([[[1.], [4.], [7.]]])
    assert_array_equal(resampled, expected)

    # Test downsampling variable length dataset
    X = to_time_series_dataset([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5]])
    resampled = TimeSeriesResampler(sz=3, method="uniform").fit_transform(X)
    expected = np.array([
        [[1.], [4.], [7.]],
        [[1.], [3.], [5.]]]
    )
    assert_array_equal(resampled, expected)

    # Test downsampling multivariate
    X = to_time_series_dataset([[[1, 2], [3, 4], [5, 6], [7, 8] , [1, 2], [3, 4]]])
    resampled = TimeSeriesResampler(sz=3, method="uniform").fit_transform(X)
    expected = np.array([[[1., 2.], [5., 6.], [3., 4.]]])
    assert_array_equal(resampled, expected)

    # Test upsampling
    X = to_time_series_dataset([1, 2, 3, 4, 5])
    resampled = TimeSeriesResampler(sz=7, method="uniform").fit_transform(X)
    expected = np.array([[[1.], [2.], [2.], [3.], [4.], [4.], [5.]]])
    assert_array_equal(resampled, expected)

def test_resampler_linear():

    # Test downsampling target_size = 1
    X = to_time_series_dataset([1, 2, 3, 4, 5, 6, 7])
    resampled = TimeSeriesResampler(sz=1, method="linear").fit_transform(X)
    expected = np.array([[[4.]]])
    assert_array_equal(resampled, expected)

    # Test downsampling
    X = to_time_series_dataset([1, 2, 3, 4, 5, 6, 7])
    resampled = TimeSeriesResampler(sz=3, method="linear").fit_transform(X)
    expected = np.array([[[1.], [4.], [7.]]])
    assert_array_equal(resampled, expected)

    # Test downsampling variable length dataset
    X = to_time_series_dataset([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5]])
    resampled = TimeSeriesResampler(sz=3, method="linear").fit_transform(X)
    expected = np.array([
        [[1.], [4.], [7.]],
        [[1.], [3.], [5.]]]
    )
    assert_array_equal(resampled, expected)

    # Test downsampling multivariate
    X = to_time_series_dataset([[[1, 2], [3, 4], [5, 6], [7, 8] , [1, 2], [3, 4]]])
    resampled = TimeSeriesResampler(sz=3, method="linear").fit_transform(X)
    expected = np.array([[[1., 2.], [6., 7.], [3., 4.]]])
    np.testing.assert_array_almost_equal(resampled, expected)

    # Test upsampling
    X = to_time_series_dataset([1, 2, 3, 4, 5])
    resampled = TimeSeriesResampler(sz=7, method="linear").fit_transform(X)
    expected = np.array([[[1.], [5/3], [7/3], [3.], [11/3], [13/3], [5.]]])
    np.testing.assert_array_almost_equal(resampled, expected)


def test_resampler_mean():

    # Test downsampling target_size = 1
    X = to_time_series_dataset([1, 2, 3, 4, 5, 6, 7])
    resampled = TimeSeriesResampler(sz=1, method="mean").fit_transform(X)
    expected = np.array([[[4.]]])
    assert_array_equal(resampled, expected)

    # Test downsampling with integer size / target_size
    X = to_time_series_dataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    resampled = TimeSeriesResampler(sz=5, method="mean").fit_transform(X)
    expected = np.array([[[1.5], [3.5], [5.5], [7.5], [9.5]]])
    assert_array_equal(resampled, expected)

    # Test downsampling with float size / target_size
    X = to_time_series_dataset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    resampled = TimeSeriesResampler(sz=7, method="mean").fit_transform(X)
    expected = np.array([[[1], [2.5], [4.], [5.5], [7.], [8.5], [10]]])
    assert_array_equal(resampled, expected)

    # Test downsampling variable length dataset
    X = to_time_series_dataset([[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6]])
    resampled = TimeSeriesResampler(sz=3, method="mean").fit_transform(X)
    expected = np.array([
        [[1.5], [4.], [6.5]],
        [[1.5], [3.5], [5.5]]]
    )
    assert_array_equal(resampled, expected)

    # Test downsampling multivariate
    X = to_time_series_dataset([[[1, 4], [3, 2], [5, 1], [7, 2], [1, 8], [3, 4]]])
    resampled = TimeSeriesResampler(sz=3, method="mean").fit_transform(X)
    expected = np.array([[[2., 3.], [6., 1.5], [2., 6.]]])
    np.testing.assert_array_almost_equal(resampled, expected)

    # Test downsampling multivariate with nans
    X = to_time_series_dataset([[[1, 4], [3, 2], [np.nan, 1], [7, 2], [1, 8], [3, 4], [7, 9]]])
    resampled = TimeSeriesResampler(sz=3, method="mean").fit_transform(X)
    expected = np.array([[[2, 3], [4., 11/3], [5, 6.5]]])
    np.testing.assert_array_almost_equal(resampled, expected)

    # Test downsampling multivariate with window and nans
    X = to_time_series_dataset([[[1, 4], [3, 2], [np.nan, 1], [7, 2], [1, 8], [3, 4], [7, 9]]])
    resampled = TimeSeriesResampler(sz=3, method="mean", window_size=4).fit_transform(X)
    expected = np.array([[[2, 7/3], [3.5, 17/5], [11/3, 7]]])
    np.testing.assert_array_almost_equal(resampled, expected)

    # Test upsampling
    X = to_time_series_dataset([1, 2, 3, 4, 5])
    resampled = TimeSeriesResampler(sz=7, method="mean").fit_transform(X)
    expected = np.array([[[1.], [1.5], [2.5], [3], [3.5], [4.5], [5.]]])
    np.testing.assert_array_almost_equal(resampled, expected)

    # Test upsampling multivariate
    X = to_time_series_dataset([[[1, 4], [3, 2], [5, 1], [7, 2], [1, 8], [3, 4]]])
    resampled = TimeSeriesResampler(sz=10, method="mean").fit_transform(X)
    expected = np.array([[
        [1, 4], [2., 3.], [3., 2.], [4, 1.5], [6., 1.5], [6, 1.5], [4, 5], [1, 8], [2, 6], [3., 4.]]]
    )
    np.testing.assert_array_almost_equal(resampled, expected)

    # Test upsampling with window
    X = to_time_series_dataset([1, 2, 3, 4, 5])
    resampled = TimeSeriesResampler(sz=7, method="mean", window_size=3).fit_transform(X)
    expected = np.array([[[1.5], [2], [2], [3], [4], [4], [4.5]]])
    np.testing.assert_array_almost_equal(resampled, expected)


def test_resampler_max():

    # Test downsampling with integer size / target_size
    X = to_time_series_dataset([1, 2, 3, 4, 5, 6])
    resampled = TimeSeriesResampler(sz=3, method="max").fit_transform(X)
    expected = np.array([
        [[2], [4], [6]]
    ])
    np.testing.assert_array_equal(resampled, expected)

    # Test downsampling with non integer size / target_size
    X = to_time_series_dataset([1, 2, 3, 4, 5, 6, 7])
    resampled = TimeSeriesResampler(sz=3, method="max").fit_transform(X)
    expected = np.array([
        [[2.], [5], [7]]
    ])
    np.testing.assert_array_equal(resampled, expected)

    # Test variable length
    X = to_time_series_dataset([[1, 2, 3, 4, 5, 6], [6, 8, 10, 12], [4, 5, 6, 7, 8, 9, 10, 11, 12]])
    resampled = TimeSeriesResampler(sz=6, method="max").fit_transform(X)
    expected = np.array([
        [[1.], [2.], [3.], [4.], [5.], [6.]],
        [[6], [8], [8], [10], [12], [12]],
        [[4.], [6.], [7.], [9.], [11.], [12.]]
    ])
    np.testing.assert_array_equal(resampled, expected)


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


def test_min_max_scaler_invalid_range():
    with pytest.raises(ValueError):
        TimeSeriesScalerMinMax((1,0)).fit_transform([[1, 2, 3]])


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
    expected = np.array([
        [[1], [1], [1], [3], [6], [6], [9] ],
        [[1], [2], [2], [9], [np.nan], [np.nan], [np.nan] ],
        [[np.nan], [2], [2], [9], [6], [6], [np.nan]],
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
    value=42.42
    imputer.set_params(value=value)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, value], [2, 3], [2, value]],
        [[3, 4], [value, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [value], [value], [3], [6], [value], [9]],
        [[1], [2], [value], [9], [np.nan], [np.nan], [np.nan]],
        [[value], [2], [value], [9], [6], [value], [np.nan]],
    ])
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
    expected = np.array([
        [[1], [5/3], [7/3], [3], [6], [7.5], [9]],
        [[1], [2], [5.5], [9], [np.nan], [np.nan], [np.nan]],
        [[2], [2], [5.5], [9], [6], [6], [np.nan]],
    ])
    np.testing.assert_array_almost_equal(transformed, expected)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, 3], [1.5, 3], [2, 3], [np.nan, np.nan]],
        [[3, 4], [4.5, 5], [6, 6], [6, 7]],
    ])
    np.testing.assert_array_almost_equal(transformed, expected)

    multivariate_dataset = [
        [[1, np.nan], [2, 3], [2, np.nan]],
        [[3, 4], [np.nan, 5], [np.nan, np.nan]],
    ]
    univariate_dataset = [
        [1, np.nan, np.nan, 3, 6, np.nan, 9],
        [1, 2, np.nan, 9],
        [np.nan, 2, np.nan, 9, 6, np.nan],
    ]
    imputer.set_params(method="constant", keep_trailing_nans=True)
    transformed = imputer.fit_transform(multivariate_dataset)
    expected = np.array([
        [[1, value], [2, 3], [2, value]],
        [[3, 4], [value, 5], [np.nan, np.nan]],
    ])
    np.testing.assert_array_equal(transformed, expected)
    transformed = imputer.fit_transform(univariate_dataset)
    expected = np.array([
        [[1], [value], [value], [3], [6], [value], [9]],
        [[1], [2], [value], [9], [np.nan], [np.nan], [np.nan]],
        [[value], [2], [value], [9], [6], [np.nan], [np.nan]],
    ])
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
