import numpy as np
import pytest

from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.utils import to_time_series_dataset


@pytest.mark.parametrize("metric", ["dtw", "euclidean"])
@pytest.mark.parametrize("radius", [0., 1., 1e20, np.inf])
@pytest.mark.parametrize("return_distance", [False, True])
@pytest.mark.parametrize("explicit_query", [False, True])
def test_radius_neighbors_excludes_only_implicit_self(
    metric, radius, return_distance, explicit_query
):
    # The duplicate is a valid neighbor even when the query itself is excluded.
    values = np.array([0., 0., 2.])
    X = to_time_series_dataset(np.repeat(values[:, None], 2, axis=1))
    original = X.copy()
    model = KNeighborsTimeSeries(
        n_neighbors=1, metric=metric, n_jobs=1
    ).fit(X)
    result = model.radius_neighbors(
        X if explicit_query else None,
        radius=radius,
        return_distance=return_distance,
        sort_results=return_distance,
    )
    if return_distance:
        distances, indices = result
    else:
        indices = result

    # Both Euclidean distance and DTW on these constant two-point series
    # equal sqrt(2) times the difference between their values.
    expected_distances = np.sqrt(2) * np.abs(values[:, None] - values)
    assert indices.shape == (len(X),)
    for i, neighbors in enumerate(indices):
        expected = expected_distances[i] <= radius
        if not explicit_query:
            expected[i] = False
        np.testing.assert_array_equal(
            np.sort(neighbors), np.flatnonzero(expected)
        )
        if return_distance:
            np.testing.assert_allclose(
                distances[i], expected_distances[i, neighbors]
            )
            assert np.all(np.diff(distances[i]) >= 0)

    # Querying must leave the fitted model and original input reusable.
    assert model.metric == metric
    assert model.n_features_in_ == 1
    np.testing.assert_array_equal(X, original)
    np.testing.assert_allclose(model.kneighbors(X)[0], 0.)
