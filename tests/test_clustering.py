import math

import numpy as np

import pytest

from scipy.spatial.distance import cdist

from tslearn.clustering import (
    EmptyClusterError,
    TimeSeriesKMeans,
    KernelKMeans,
    KShape,
    TimeSeriesDBSCAN
)
from tslearn.clustering.utils import (
    _check_full_length,
    _check_no_empty_cluster,
    silhouette_score
)
from tslearn.generators import random_walks
from tslearn.metrics import cdist_dtw, cdist_soft_dtw, dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset, ts_size


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_check_no_empty_cluster():
    labels = np.array([1, 1, 2, 0, 2])
    _check_no_empty_cluster(labels, 3)
    np.testing.assert_raises(EmptyClusterError, _check_no_empty_cluster,
                             labels, 4)


def test_check_full_length():
    centroids = to_time_series_dataset([[1, 2, 3], [1, 2, 3, 4, 5]])
    arr = _check_full_length(centroids)
    np.testing.assert_allclose(arr,
                               to_time_series_dataset([[1, 1.5, 2, 2.5, 3],
                                                       [1, 2, 3, 4, 5]]))


def test_kernel_kmeans():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)

    gak_km = KernelKMeans(n_clusters=3, verbose=False,
                          max_iter=5,
                          random_state=rng).fit(time_series)
    np.testing.assert_allclose(gak_km.labels_, gak_km.predict(time_series))

    gak_km = KernelKMeans(n_clusters=101, verbose=False,
                          max_iter=5,
                          random_state=rng).fit(time_series)
    assert gak_km._X_fit is None

    with pytest.raises(RuntimeError):
        KernelKMeans(n_clusters=101, verbose=False,
                     max_iter=5,
                     kernel_params={"sigma": 0},
                     random_state=rng).fit(time_series)

    gak_km = KernelKMeans(n_clusters=2, verbose=False, kernel="rbf",
                          kernel_params={"gamma": 1.},
                          max_iter=5,
                          random_state=rng).fit(time_series)
    assert gak_km.sigma_gak_ is None


def test_kmeans():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)

    km = TimeSeriesKMeans(n_clusters=3, metric="euclidean", max_iter=5,
                          verbose=False, random_state=rng).fit(time_series)
    dists = cdist(time_series.reshape((n, -1)),
                  km.cluster_centers_.reshape((3, -1)))
    np.testing.assert_allclose(km.labels_, dists.argmin(axis=1))
    np.testing.assert_allclose(km.labels_, km.predict(time_series))

    km_dba = TimeSeriesKMeans(n_clusters=3,
                              metric="dtw",
                              max_iter=5,
                              verbose=False,
                              random_state=rng).fit(time_series)
    dists = cdist_dtw(time_series, km_dba.cluster_centers_)
    np.testing.assert_allclose(km_dba.labels_, dists.argmin(axis=1))
    np.testing.assert_allclose(km_dba.labels_, km_dba.predict(time_series))

    km_sdtw = TimeSeriesKMeans(n_clusters=3,
                               metric="softdtw",
                               max_iter=5,
                               verbose=False,
                               random_state=rng).fit(time_series)
    dists = cdist_soft_dtw(time_series, km_sdtw.cluster_centers_)
    np.testing.assert_allclose(km_sdtw.labels_, dists.argmin(axis=1))
    np.testing.assert_allclose(km_sdtw.labels_, km_sdtw.predict(time_series))

    km_nofit = TimeSeriesKMeans(n_clusters=101,
                                verbose=False,
                                random_state=rng).fit(time_series)
    assert(km_nofit._X_fit is None)

    with pytest.raises(ValueError):
        KShape(n_clusters=101, verbose=False, init="random").fit(time_series)

    with pytest.raises(ValueError):
        KShape(n_clusters=2, verbose=False, init="invalid").fit(time_series)

    X_bis = to_time_series_dataset([[1, 2, 3, 4],
                                    [1, 2, 3],
                                    [2, 5, 6, 7, 8, 9]])
    TimeSeriesKMeans(n_clusters=2, verbose=False, max_iter=5,
                     metric="softdtw", random_state=0).fit(X_bis)
    TimeSeriesKMeans(n_clusters=2, verbose=False, max_iter=5,
                     metric="dtw", random_state=0,
                     init="random").fit(X_bis)
    TimeSeriesKMeans(n_clusters=2, verbose=False, max_iter=5,
                     metric="dtw", random_state=0,
                     init="k-means++").fit(X_bis)
    TimeSeriesKMeans(n_clusters=2, verbose=False, max_iter=5,
                     metric="dtw", init=X_bis[:2]).fit(X_bis)

    # Barycenter size (nb of timestamps)
    # Case 1. kmeans++ / random init
    n, sz, d = 15, 10, 1
    n_clusters = 3
    time_series = rng.randn(n, sz, d)

    sizes_all_same_series = [sz] * n_clusters
    km_euc = TimeSeriesKMeans(n_clusters=3,
                              metric="euclidean",
                              max_iter=5,
                              verbose=False,
                              init="k-means++",
                              random_state=rng).fit(time_series)
    np.testing.assert_equal(sizes_all_same_series,
                            [ts_size(b) for b in km_euc.cluster_centers_])
    km_dba = TimeSeriesKMeans(n_clusters=3,
                              metric="dtw",
                              max_iter=5,
                              verbose=False,
                              init="random",
                              random_state=rng).fit(time_series)
    np.testing.assert_equal(sizes_all_same_series,
                            [ts_size(b) for b in km_dba.cluster_centers_])

    # Case 2. forced init
    barys = to_time_series_dataset([[1., 2., 3.],
                                    [1., 2., 2., 3., 4.],
                                    [3., 2., 1.]])
    sizes_all_same_bary = [barys.shape[1]] * n_clusters
    # If Euclidean is used, barycenters size should be that of the input series
    km_euc = TimeSeriesKMeans(n_clusters=3,
                              metric="euclidean",
                              max_iter=5,
                              verbose=False,
                              init=barys,
                              random_state=rng)
    np.testing.assert_raises(ValueError, km_euc.fit, time_series)

    km_dba = TimeSeriesKMeans(n_clusters=3,
                              metric="dtw",
                              max_iter=5,
                              verbose=False,
                              init=barys,
                              random_state=rng).fit(time_series)
    np.testing.assert_equal(sizes_all_same_bary,
                            [ts_size(b) for b in km_dba.cluster_centers_])
    km_sdtw = TimeSeriesKMeans(n_clusters=3,
                               metric="softdtw",
                               max_iter=5,
                               verbose=False,
                               init=barys,
                               random_state=rng).fit(time_series)
    np.testing.assert_equal(sizes_all_same_bary,
                            [ts_size(b) for b in km_sdtw.cluster_centers_])

    # A simple dataset, can we extract the correct number of clusters?
    time_series = to_time_series_dataset([[1, 2, 3],
                                   [7, 8, 9, 11],
                                   [.1, .2, 2.],
                                   [1, 1, 1, 9],
                                   [10, 20, 30, 1000]])
    preds = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5,
                             random_state=rng).fit_predict(time_series)
    np.testing.assert_equal(set(preds), set(range(3)))
    preds = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=5,
                             random_state=rng).fit_predict(time_series)
    np.testing.assert_equal(set(preds), set(range(4)))


def test_kshape():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    time_series = TimeSeriesScalerMeanVariance().fit_transform(time_series)

    ks = KShape(n_clusters=3, n_init=1, verbose=False,
                random_state=rng).fit(time_series)
    dists = ks._cross_dists(time_series)
    np.testing.assert_allclose(ks.labels_, dists.argmin(axis=1))
    np.testing.assert_allclose(ks.labels_, ks.predict(time_series))

    with pytest.raises(ValueError):
        KShape(n_clusters=101, verbose=False, random_state=rng).fit(time_series)

    with pytest.raises(ValueError):
        KShape(n_clusters=2, verbose=False, init="invalid").fit(time_series)

    # Test that shape extraction operates on second features
    feature_1 = rng.randn(1, 10, 1)
    feature_2_0 = rng.randn(1, 10, 1) + 10
    feature_2_1 = rng.randn(1, 10, 1) - 10
    X1 = np.dstack((feature_1, feature_2_0))
    X2 = np.dstack((feature_1, feature_2_1))
    X = np.vstack((
        np.repeat(X1, 10, axis=0),
        np.repeat(X2, 10, axis=0),
    ))

    X = TimeSeriesScalerMeanVariance().fit_transform(X)
    kshape = KShape(n_clusters=2, n_init=5, random_state=rng).fit(X)
    assert all(kshape.labels_[0] == kshape.labels_[:10])
    assert all(kshape.labels_[10] == kshape.labels_[10:])
    assert kshape.labels_[0] != kshape.labels_[10]


def test_silhouette():
    np.random.seed(0)
    X = random_walks(n_ts=20, sz=16, d=1)
    labels = np.random.randint(2, size=20)
    assert math.isclose(
        silhouette_score(X, labels, metric="dtw"),
        0.13383800,
        rel_tol=1e-07
    )
    assert math.isclose(
        silhouette_score(X, labels, metric=dtw),
        0.13383800,
        rel_tol=1e-07
    )
    assert math.isclose(
        silhouette_score(cdist_dtw(X), labels, metric="precomputed"),
        0.13383800,
        rel_tol=1e-07
    )
    assert math.isclose(
        silhouette_score(X, labels, metric="euclidean"),
        0.09126917,
        rel_tol=1e-07
    )
    assert math.isclose(
        silhouette_score(X, labels, metric="softdtw"),
        0.17953934,
        rel_tol=1e-07
    )


def test_dbscan():
    # Basic clustering
    X = np.vstack((
        np.eye(3).reshape(-1, 3),
        -1 * np.eye(3).reshape(-1, 3)
    ))
    X = np.insert(X, 0, 0, axis=1)
    X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)
    X = to_time_series_dataset(X)

    db = TimeSeriesDBSCAN(eps=1e-6, min_ts=3)

    # Test invalid metric
    db.set_params(metric='gak')
    with pytest.raises(ValueError, match="Metric must be one of"):
        db.fit(X)

    # Test TSlearn metrics
    metrics = ['dtw', 'ctw', 'frechet']
    for metric in metrics:
        db.set_params(metric=metric)
        db.fit(X)
        np.testing.assert_equal(db.labels_, [0, 0, 0, 1, 1, 1])
        np.testing.assert_equal(db.components_, X)
        np.testing.assert_equal(db.core_ts_indices_, np.arange(X.shape[0]))

    # Euclidean, no clustering performed
    db.set_params(metric='euclidean')
    db.fit(X)
    np.testing.assert_equal(db.labels_, [-1, -1, -1, -1, -1, -1])
    np.testing.assert_equal(db.components_, np.array([]).reshape((0, 5)))
    np.testing.assert_equal(db.core_ts_indices_, np.array([]))

    # Test precomputed
    db.set_params(metric='precomputed')
    db.fit(cdist_dtw(X))
    np.testing.assert_equal(db.labels_, [0, 0, 0, 1, 1, 1])
    np.testing.assert_equal(db.core_ts_indices_, np.arange(X.shape[0]))

    # Clustering with outliers
    X = np.append(X, np.array([[0], [1.5], [0], [0], [0]])).reshape(-1, 5)
    X = to_time_series_dataset(X)
    db = TimeSeriesDBSCAN(eps=1e-6, min_ts=3)
    db.fit(X)
    np.testing.assert_equal(db.labels_, [0, 0, 0, 1, 1, 1, -1])
    np.testing.assert_equal(db.components_, X[:-1])
    np.testing.assert_equal(db.core_ts_indices_, np.arange(X.shape[0] - 1))

    # Check eps: increase eps so that last point is clustered
    db.set_params(eps=0.5)
    db.fit(X)
    np.testing.assert_equal(db.labels_, [0, 0, 0, 1, 1, 1, 0])
    np.testing.assert_equal(db.components_, X)
    np.testing.assert_equal(db.core_ts_indices_, np.arange(X.shape[0]))

    # Check min_ts: last point only has 1 neighboor within the eps range.
    # Therefore, it is not considered a core component
    X[0, 1, 0] = 0.9
    X[1, 2, 0] = 0.9
    db.fit(X)
    np.testing.assert_equal(db.labels_, [0, 0, 0, 1, 1, 1, 0])
    np.testing.assert_equal(db.components_, X[:-1])
    np.testing.assert_equal(db.core_ts_indices_, np.arange(X.shape[0] -1))

    # Check nb_jobs
    db = TimeSeriesDBSCAN(metric_params={'n_jobs': 1})
    assert db._get_metric_params() == {'n_jobs': 1}
    db = TimeSeriesDBSCAN(n_jobs=5, metric_params={'n_jobs': 1})
    assert db._get_metric_params() == {'n_jobs': 5}
