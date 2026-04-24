import numpy as np

import pytest

from tslearn.early_classification import NonMyopicEarlyClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset


def test_NonMyopicEarlyClassifier():

    dataset = to_time_series_dataset(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [3, 2, 1, 1, 2, 3],
            [3, 2, 1, 1, 2, 3],
        ]
    )

    y = [0, 0, 0, 1, 1, 1, 0, 0]
    model = NonMyopicEarlyClassifier(
        n_clusters=3,
        base_classifier=KNeighborsTimeSeriesClassifier(
            n_neighbors=1, metric="euclidean"
        ),
        min_t=2,
        lamb=1000.0,
        cost_time_parameter=0.1,
        random_state=0,
    )
    assert model.classes_ is None
    model.fit(dataset, y)
    np.testing.assert_almost_equal(model.early_classification_cost(dataset, y), 0.35)

    # Fewer timestamps than min_ts
    pred, delays = model.early_predict(dataset[:, :1])
    np.testing.assert_array_equal(pred, np.array([np.nan] * 8))
    np.testing.assert_array_equal(delays, np.array([np.nan] * 8))

    pred, delays = model.early_predict(dataset[:, :3])
    np.testing.assert_array_equal(pred, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(delays, np.array([1, 1, 1, 1, 1, 1, 0, 0]))

    # More timestamps than trained dataset
    data = to_time_series_dataset([[1, 2, 3, 3, 2, 1, 1, 2, 3]])
    with pytest.raises(ValueError):
        model.early_predict(data)
