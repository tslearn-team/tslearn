import numpy as np

import pytest

from tslearn.preprocessing import TimeSeriesFeatureSynchronizer


def test_TimeSeriesFeatureSynchronizer():
    synchronizer = TimeSeriesFeatureSynchronizer()

    data = [
        [[1, 2], [2, np.nan]],
        [[1, 2], [np.nan, 3]],
    ]
    transformed = synchronizer.fit_transform(data)
    expected = np.array([[[1., 2.], [2., 2.]], [[1., 2.], [np.nan, np.nan]]])
    np.testing.assert_array_equal(transformed, expected)

    data = [
        [[1, 2], [2, 4] , [9, np.nan]],
    ]

    timestamps = np.array([
        [np.array(["2025-01-01", "2025-01-02"], dtype='datetime64'),
         np.array(["2025-01-03", "2025-01-07"], dtype='datetime64'),
         np.array(["2025-01-10", "nat"], dtype='datetime64')],
    ])
    transformed = synchronizer.fit_transform(data, timestamps=timestamps)
    expected = np.array([[[1, 2], [2, 2.4] , [9, 4]]])
    np.testing.assert_array_equal(transformed, expected)

    synchronizer.reference_feature_index = 1
    transformed = synchronizer.fit_transform(data, timestamps=timestamps)
    expected = np.array([[[1.5, 2], [6, 4]]])
    np.testing.assert_array_equal(transformed, expected)

    # Timestamps should be increasing
    timestamps = np.array([
        [np.array(["2025-01-01", "2025-01-02"], dtype='datetime64'),
         np.array(["2025-01-03", "2024-01-04"], dtype='datetime64'),
         np.array(["2025-01-05", "nat"], dtype='datetime64')],
    ])
    with pytest.raises(ValueError):
        synchronizer.fit_transform(data, timestamps=timestamps)

    # dataset and timestamps should be of same shape
    timestamps = np.array([
        [np.array(["2025-01-01", "2025-01-02"], dtype='datetime64'),
         np.array(["2025-01-03", "2025-01-04"], dtype='datetime64')]
    ])
    with pytest.raises(ValueError):
        synchronizer.fit_transform(data, timestamps=timestamps)


@pytest.mark.parametrize("origin", ["1960-01-01", "1970-01-01", "2025-01-01"])
@pytest.mark.parametrize("unit", ["us", "ns"])
@pytest.mark.parametrize("reference_feature_index", [0, 1])
@pytest.mark.parametrize("method", ["fit_transform", "transform"])
def test_feature_synchronizer_timestamp_precision(
    origin, unit, reference_feature_index, method
):
    # The second feature is linear: value 15 at tick 3, value 30 at tick 6.
    # On reference feature 0's grid (ticks 0, 2, 4) it must be 0, 10, 20,
    # independently of the absolute acquisition date.
    data = np.array([[[1., 0.], [2., 15.], [3., 30.]]])
    offsets = np.array([[[0, 0], [2, 3], [4, 6]]])
    timestamps = (
        np.datetime64(origin, unit)
        + offsets.astype("timedelta64[" + unit + "]")
    )
    data_before = data.copy()
    timestamps_before = timestamps.copy()
    synchronizer = TimeSeriesFeatureSynchronizer(reference_feature_index)
    if method == "transform":
        synchronizer.fit(data)
    actual = getattr(synchronizer, method)(data, timestamps=timestamps)
    if reference_feature_index == 0:
        expected = [[[1., 0.], [2., 10.], [3., 20.]]]
    else:
        expected = [[[1., 0.], [2.5, 15.], [3., 30.]]]
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)
    np.testing.assert_array_equal(data, data_before)
    np.testing.assert_array_equal(timestamps, timestamps_before)


@pytest.mark.parametrize("origin", ["1970-01-01", "2025-01-01"])
def test_feature_synchronizer_timestamp_precision_with_padding(origin):
    data = np.array([
        [[1., 0.], [2., 15.], [3., 30.], [np.nan, 45.]],
        [[1., 0.], [2., 15.], [3., 30.], [4., np.nan]],
    ])
    offsets = np.array([
        [[0, 0], [2, 3], [4, 6], [0, 9]],
        [[0, 0], [2, 3], [4, 6], [6, 0]],
    ])
    timestamps = (
        np.datetime64(origin, "ns") + offsets.astype("timedelta64[ns]")
    )
    timestamps[np.isnan(data)] = np.datetime64("NaT", "ns")
    actual = TimeSeriesFeatureSynchronizer().fit_transform(
        data, timestamps=timestamps
    )
    expected = [
        [[1., 0.], [2., 10.], [3., 20.], [np.nan, np.nan]],
        [[1., 0.], [2., 10.], [3., 20.], [4., 30.]],
    ]
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_feature_synchronizer_wide_timestamp_range():
    # These valid ns timestamps span more than an int64 timedelta can hold.
    dates = [["1678-01-01", "1678-01-01"],
             ["1970-01-01", "1970-01-02"],
             ["2261-01-01", "2261-01-01"]]
    timestamps = np.array([dates], dtype="datetime64[ns]")
    data = np.array([[[1., 0.], [2., 1.], [3., 2.]]])
    actual = TimeSeriesFeatureSynchronizer().fit_transform(
        data, timestamps=timestamps
    )
    # Independently compute the middle sample in whole days, without ns
    # subtraction or the transformer's interpolation implementation.
    first = np.datetime64("1678-01-01", "D")
    numerator = int((np.datetime64("1970-01-01", "D") - first).astype("int64"))
    denominator = int(
        (np.datetime64("1970-01-02", "D") - first).astype("int64")
    )
    expected = [[[1., 0.], [2., numerator / denominator], [3., 2.]]]
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_feature_synchronizer_fine_interval_after_wide_span():
    # A single global floating origin still loses the final 1 ns interval.
    first = np.datetime64("1678-01-01", "ns")
    middle = np.datetime64("1970-01-01", "ns")
    last = np.datetime64("2261-01-01", "ns")
    timestamps = np.array([[
        [first, first],
        [middle, middle],
        [last + np.timedelta64(1, "ns"), last],
        [last + np.timedelta64(2, "ns"), last + np.timedelta64(2, "ns")],
    ]])
    data = np.array([[[1., 0.], [2., 1.], [3., 10.], [4., 20.]]])
    actual = TimeSeriesFeatureSynchronizer().fit_transform(
        data, timestamps=timestamps
    )
    expected = [[[1., 0.], [2., 1.], [3., 15.], [4., 20.]]]
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize("reference_feature_index", [0, 1])
@pytest.mark.parametrize("near_left", [False, True])
@pytest.mark.parametrize("method", ["fit_transform", "transform"])
def test_feature_synchronizer_timestamp_precision_unequal_intervals(
    reference_feature_index, near_left, method
):
    # A 1 ns displacement in a 10**16 ns interval contributes 1e-10.
    # Adding its fractional position to a float knot index would lose it.
    query = 1 if near_left else -1
    values = [0., 0., 1e6] if near_left else [1e6, 0., 0.]
    data = np.stack([np.arange(3, dtype=float), values], axis=-1)[None]
    ticks = np.array([[[-10**16, -10**16],
                       [query, 0], [10**16, 10**16]]])
    expected = np.array([[[0., values[0]], [1., 1e-10], [2., values[2]]]])
    if reference_feature_index == 1:
        data = data[..., ::-1]
        ticks = ticks[..., ::-1]
        expected = expected[..., ::-1]
    timestamps = ticks.astype("datetime64[ns]")
    synchronizer = TimeSeriesFeatureSynchronizer(reference_feature_index)
    if method == "transform":
        synchronizer.fit(data)
    actual = getattr(synchronizer, method)(data, timestamps=timestamps)
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=0)


@pytest.mark.parametrize("values", [
    [np.inf, np.inf], [np.inf, 1.], [1., np.inf], [-np.inf, np.inf],
    [-1e308, 1e308], [1., np.nan, 3.],
])
def test_feature_synchronizer_timestamp_precision_nonfinite_values(values):
    # Keep the existing np.interp behavior for supported nonfinite inputs
    # and finite data whose value difference overflows.
    size = len(values)
    queries = np.arange(size) * 2 + 1
    points = np.arange(size) * 4
    data = np.stack([np.arange(size, dtype=float), values], axis=-1)[None]
    timestamps = np.stack([queries, points], axis=-1)[None].astype(
        "datetime64[ns]"
    )
    actual = TimeSeriesFeatureSynchronizer().fit_transform(
        data, timestamps=timestamps
    )
    expected = np.interp(queries, points, values)
    np.testing.assert_array_equal(actual[0, :, 1], expected)
