import numpy as np

import pytest

from tslearn.neighbors import KNeighborsTimeSeries, KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_k_neighbors_timeseries():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)

    model = KNeighborsTimeSeries()
    np.testing.assert_equal(
        model.fit(X).kneighbors(X, return_distance=False)[0],
        [0, 13, 7, 12, 3]
    )

    model = KNeighborsTimeSeries(metric='ctw')
    np.testing.assert_equal(
        model.fit(X).kneighbors(X, return_distance=False)[0],
        [0, 13, 7, 12, 3]
    )

    model = KNeighborsTimeSeries(metric='softdtw')
    np.testing.assert_equal(
        model.fit(X).kneighbors(X, return_distance=False)[0],
        [0, 13, 12, 7, 3]
    )

    model = KNeighborsTimeSeries(metric='frechet')
    np.testing.assert_equal(
        model.fit(X).kneighbors(X, return_distance=False)[0],
        [0, 3, 13, 5, 1]
    )

    with pytest.raises(ValueError):
        KNeighborsTimeSeries(metric='invalid').fit(X)

    with pytest.raises(ValueError):
        KNeighborsTimeSeries(metric='lcss').fit(X).kneighbors()

    with pytest.raises(ValueError):
        KNeighborsTimeSeries().fit(X).kneighbors(X, n_neighbors=len(X) + 1)

    with pytest.raises(ValueError):
        KNeighborsTimeSeries().fit(X).kneighbors(n_neighbors=len(X))


@pytest.mark.parametrize("n_neighbors", [1, 4])
@pytest.mark.parametrize("metric", [
    'dtw', 'softdtw', 'ctw', 'euclidean', 'sqeuclidean', 'cityblock', 'sax'
])
def test_kneighbors_excludes_self(metric, n_neighbors):
    rng = np.random.RandomState(0)
    X = rng.randn(5, 10, 3)

    # Ensure that a TS is not considered a neighbor of itself
    model = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=metric).fit(X)
    distances, indices = model.kneighbors()
    assert indices.shape == (len(X), n_neighbors)
    assert np.all(indices != np.arange(len(X))[:, None])

    # Same with duplicates (several 0 distance)
    X[-1], X[-2] = X[0], X[1]
    model.fit(X)
    distances, indices = model.kneighbors()
    assert indices.shape == (len(X), n_neighbors)
    assert np.all(indices != np.arange(len(X))[:, None])


@pytest.mark.parametrize("n_neighbors", [1, 4])
@pytest.mark.parametrize("metric", [
    'dtw', 'softdtw', 'ctw', 'euclidean', 'sqeuclidean', 'cityblock', 'sax'
])
def test_kneighbors_radius_neighbors(metric, n_neighbors):
    rng = np.random.RandomState(0)
    X = rng.randn(5, 10, 3)

    model = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=metric).fit(X)

    if metric == "softdtw":
       with pytest.raises(ValueError):
           model.radius_neighbors()
       return

    # Zeroish radius, query same as fitted data
    distances, neighbors = model.radius_neighbors(X, radius=0.1)
    for i in range(len(X)):
        assert distances.shape == neighbors.shape == (len(X),)
        np.testing.assert_array_almost_equal(distances[i], [0])
        np.testing.assert_array_equal(neighbors[i], [i])

    # Non Zero radius, on fitted data
    indices = model.radius_neighbors(radius=1, return_distance=False)
    assert all(i not in indices[i] for i in range(len(indices)))

    # Check kneighbors / radius_neighbors consistency
    distances, indices = model.kneighbors()
    min_dist = np.min(distances)
    min_ind1, min_ind2 = indices[distances==min_dist]  # Only 2 indices in our case
    distances_, neighbors = model.radius_neighbors(radius=min_dist + 0.01)
    for i in range(len(X)):
        if i == min_ind1:
            assert distances_[i].shape[-1] == neighbors[i].shape[-1] ==  1
            assert distances_[i][0] == min_dist
            np.testing.assert_array_equal(neighbors[i], [min_ind2])
        elif i ==min_ind2:
            assert distances_[i].shape[-1] == neighbors[i].shape[-1] == 1
            assert distances_[i][0] == min_dist
            np.testing.assert_array_equal(neighbors[i], [min_ind1])
        else:
            assert distances[i].ndim == 1


def test_kneighbors_sax():
    X = to_time_series_dataset([[-3., -2., -1., -2.], [1., 2., 3., 2.], [0, 0, 0, 0]])
    model = KNeighborsTimeSeries(
        n_neighbors=1,
        metric="sax",
        metric_params={"n_segments": 2, "alphabet_size_avg": 5, "scale": True},
    ).fit(X)
    np.testing.assert_equal(model.kneighbors(return_distance=False), np.array([[2], [2], [0]]))
    np.testing.assert_equal(model.kneighbors(X[:1]), np.array([[[0.]], [[0]]]))
    np.testing.assert_equal(model.kneighbors(X[1:2]), np.array([[[0.]], [[1]]]))
    np.testing.assert_equal(model.kneighbors(X[2:]), np.array([[[0.]], [[2]]]))

    # Check fit reinitialize sax
    X = X + 10
    model.fit(X)
    np.testing.assert_equal(model.kneighbors(return_distance=False), np.array([[2], [2], [0]]))
    np.testing.assert_equal(model.kneighbors(X[:1]), np.array([[[0.]], [[0]]]))
    np.testing.assert_equal(model.kneighbors(X[1:2]), np.array([[[0.]], [[1]]]))
    np.testing.assert_equal(model.kneighbors(X[2:]), np.array([[[0.]], [[2]]]))


def test_kneighbors_metric_params():
    rng = np.random.RandomState(0)
    X = rng.randn(10, 5, 3)
    n_neighbors = 2

    model_1 = KNeighborsTimeSeries(n_neighbors=n_neighbors,
                                   metric="euclidean").fit(X)
    model_2 = KNeighborsTimeSeries(n_neighbors=n_neighbors,
                                   metric="dtw",
                                   metric_params={"sakoe_chiba_radius": 0}).fit(X)
    model_3 = KNeighborsTimeSeries(n_neighbors=n_neighbors,
                                   metric="softdtw",
                                   metric_params={"gamma": 0,
                                                  "sakoe_chiba_radius": 0}).fit(X)
    model_4 = KNeighborsTimeSeries(n_neighbors=n_neighbors,
                                   metric="sax",
                                   metric_params={"n_segments": 5,
                                                  "alphabet_size_avg": 100}).fit(X)
    np.testing.assert_almost_equal(model_1.kneighbors(return_distance=True),
                                   model_2.kneighbors(return_distance=True))
    np.testing.assert_almost_equal(model_1.kneighbors(return_distance=True),
                                   model_3.kneighbors(return_distance=True))
    np.testing.assert_almost_equal(model_1.kneighbors(return_distance=False),
                                   model_4.kneighbors(return_distance=False))

    model = KNeighborsTimeSeries(metric_params={
        "n_jobs": "whatever",
        "verbose": "whatever"}).fit(X)
    metric_params = model._get_metric_params()
    assert "n_jobs" not in metric_params
    assert "verbose" not in metric_params
    model.kneighbors()


def test_k_neighbors_classifier():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    y = rng.randint(low=0, high=3, size=n)

    model_euc = KNeighborsTimeSeriesClassifier(n_neighbors=3,
                                               metric="euclidean")
    y_pred_euc = model_euc.fit(X, y).predict(X)
    model_dtw_sakoe = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="dtw",
        metric_params={
            "global_constraint": "sakoe_chiba",
            "sakoe_chiba_radius": 0
        }
    )
    y_pred_sakoe = model_dtw_sakoe.fit(X, y).predict(X)
    np.testing.assert_equal(y_pred_euc, y_pred_sakoe)

    model_softdtw = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="softdtw",
        metric_params={
            "gamma": 1e-6
        }
    )
    y_pred_softdtw = model_softdtw.fit(X, y).predict(X)

    model_dtw = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="dtw"
    )
    y_pred_dtw = model_dtw.fit(X, y).predict(X)

    np.testing.assert_equal(y_pred_dtw, y_pred_softdtw)

    model_ctw = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="ctw"
    )
    # Just testing that things run, nothing smart here :(
    model_ctw.fit(X, y).predict(X)

    model_sax = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="sax",
        metric_params={
            "alphabet_size_avg": 6,
            "n_segments": 10
        }
    )
    model_sax.fit(X, y)

    model_frechet = KNeighborsTimeSeriesClassifier(
        n_neighbors=1,
        metric="frechet"
    )
    np.testing.assert_equal(model_frechet.fit(X, y).predict(X), y)

    model_frechet = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="frechet"
    )
    np.testing.assert_equal(
        model_frechet.fit(X, y).kneighbors(X, return_distance=False)[0],
        [0,  3, 13]
    )

    with pytest.raises(ValueError):
        KNeighborsTimeSeriesClassifier(metric='invalid').fit(X, y)

    # The MINDIST of SAX is a lower bound of the euclidean distance
    euc_dist, _ = model_euc.kneighbors(X, n_neighbors=5)
    sax_dist, _ = model_sax.kneighbors(X, n_neighbors=5)

    # First column will contain zeroes
    np.testing.assert_array_less(sax_dist[:, 1:], euc_dist[:, 1:])
