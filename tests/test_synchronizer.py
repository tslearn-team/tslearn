import numpy as np

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
        [[1, 2], [2, 4] , [3, np.nan]],
    ]

    timestamps = np.array([
        [np.array(["2025-01-01", "2025-01-02"], dtype='datetime64'),
         np.array(["2025-01-03", "2025-01-04"], dtype='datetime64'),
         np.array(["2025-01-05", "nat"], dtype='datetime64')],
    ])
    transformed = synchronizer.fit_transform(data, timestamps=timestamps)
    expected =np.array([[[1, 2], [2, 3] , [3, 4]]])
    np.testing.assert_array_equal(transformed, expected)

    synchronizer.reference_feature_index = 1
    transformed = synchronizer.fit_transform(data, timestamps)
    expected = np.array([[[1, 2], [3, 4]]])
    np.testing.assert_array_equal(transformed, expected)