import math
import warnings

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
    silhouette_score,
    silhouette_samples
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

    gak_km = KernelKMeans(n_clusters=3,
                          verbose=False,
                          max_iter=5,
                          kernel_params={"sigma": "auto"},
                          random_state=0).fit(time_series)
    np.testing.assert_allclose(gak_km.labels_, gak_km.predict(time_series))
    assert np.isclose(gak_km.sigma_gak_, 6.7384149084372655)

    gak_km = KernelKMeans(n_clusters=3,
                          verbose=False,
                          max_iter=5,
                          random_state=rng).fit(time_series)
    np.testing.assert_allclose(gak_km.labels_, gak_km.predict(time_series))

    gak_km = KernelKMeans(n_clusters=101,
                          verbose=False,
                          max_iter=5,
                          random_state=rng).fit(time_series)
    assert gak_km._X_fit is None

    with pytest.raises(RuntimeError):
        KernelKMeans(n_clusters=101,
                     verbose=False,
                     max_iter=5,
                     kernel_params={"sigma": 0},
                     random_state=rng).fit(time_series)

    gak_km = KernelKMeans(n_clusters=2,
                          verbose=False,
                          kernel="rbf",
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
    expected_inertia = np.sum(
        np.fromiter(
            (dists[i, km.labels_[i]] ** 2 for i in range(n)),
            dtype=float
        )
    ) / n

    assert km.inertia_ == expected_inertia

    km_dtw_inertia = TimeSeriesKMeans(
        n_clusters=3,
        metric="euclidean",
        max_iter=5,
        verbose=False,
        dtw_inertia=True,
        random_state=rng
    ).fit(time_series)
    dists = cdist_dtw(time_series, km_dtw_inertia.cluster_centers_)
    expected_inertia = np.sum(
        np.fromiter(
            (dists[i, km_dtw_inertia.labels_[i]]**2 for i in range(n)),
            dtype=float
        )
    ) / n
    assert km_dtw_inertia.inertia_ == expected_inertia
    assert km.inertia_ >= km_dtw_inertia.inertia_

    assert km_dtw_inertia.inertia_ < km.inertia_

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


def test_silhouette_samples():
    np.random.seed(0)
    X = random_walks(n_ts=20, sz=16, d=1)
    labels = np.random.randint(2, size=20)
    for metric, expected in [
        ("dtw", 0.13383800),
        (dtw, 0.13383800),
        ("precomputed", 0.13383800),
        ("euclidean", 0.09126917),
        ("softdtw", 0.17953934)
    ]:
        if metric == "precomputed":
            samples = silhouette_samples(cdist_dtw(X), labels,
                                         metric="precomputed")
        else:
            samples = silhouette_samples(X, labels, metric=metric)
        assert samples.shape == (20,)
        assert np.all((samples >= -1.) & (samples <= 1.))
        assert math.isclose(float(np.mean(samples)), expected, rel_tol=1e-07)
    # sample_size / random_state are not special-cased here (unlike
    # silhouette_score): they are forwarded like any other kwarg, mirroring
    # sklearn's own silhouette_score/silhouette_samples pair. For metrics
    # whose cross-distance function has a fixed signature (dtw, softdtw, a
    # callable), an unsupported kwarg raises naturally from that call.
    for metric in ["dtw", "softdtw", dtw]:
        for bad_kw in [{"sample_size": 10}, {"random_state": 0}]:
            with pytest.raises(TypeError):
                silhouette_samples(X, labels, metric=metric, **bad_kw)
    # euclidean ignores unsupported kwargs, matching sklearn's own
    # silhouette_samples with metric="euclidean".
    silhouette_samples(X, labels, metric="euclidean", sample_size=10)
    with pytest.raises(TypeError):
        silhouette_samples(cdist_dtw(X), labels, metric="precomputed",
                           random_state=0)
    # unlike silhouette_score, an explicit None value is not special-cased
    # either: it is forwarded like any other value and still raises for
    # dtw/softdtw/a callable, since the underlying call has no such
    # parameter regardless of its value
    with pytest.raises(TypeError):
        silhouette_samples(X, labels, metric="dtw", sample_size=None,
                           random_state=None)
    # ...but is a no-op for euclidean, same as any other value
    silhouette_samples(X, labels, metric="euclidean", sample_size=None,
                       random_state=None)
    with pytest.raises(TypeError):
        silhouette_samples(cdist_dtw(X), labels, metric="precomputed",
                           sample_size=None, random_state=None)
    for bad_kw in [
        {"metric_params": {}},
        {"n_jobs": 1},
        {"verbose": 1},
    ]:
        with pytest.raises(TypeError):
            silhouette_samples(cdist_dtw(X), labels, metric="precomputed",
                               **bad_kw)
    # metric kwargs passed directly via **kwds are forwarded to the metric
    constrained = silhouette_samples(X, labels, metric="dtw",
                                     global_constraint="sakoe_chiba",
                                     sakoe_chiba_radius=2)
    unconstrained = silhouette_samples(X, labels, metric="dtw")
    assert not np.allclose(constrained, unconstrained)
    # n_jobs inside metric_params is stripped and handled by the n_jobs arg
    samples_njobs = silhouette_samples(X, labels, metric="dtw",
                                       metric_params={"n_jobs": 1})
    assert samples_njobs.shape == (20,)
    assert math.isclose(float(np.mean(samples_njobs)), 0.13383800,
                        rel_tol=1e-07)
    # metric_params are forwarded to a callable metric

    def sakoe_dtw(x, y, sakoe_chiba_radius=0):
        if sakoe_chiba_radius == 0:
            return dtw(x, y)
        return dtw(x, y, global_constraint="sakoe_chiba",
                   sakoe_chiba_radius=sakoe_chiba_radius)

    constrained = silhouette_samples(
        X, labels, metric=sakoe_dtw,
        metric_params={"sakoe_chiba_radius": 3})
    unconstrained = silhouette_samples(X, labels, metric=sakoe_dtw)
    assert not np.allclose(constrained, unconstrained)


def test_silhouette_samples_softdtw_njobs_verbose(monkeypatch):
    np.random.seed(0)
    X = random_walks(n_ts=20, sz=16, d=1)
    labels = np.random.randint(2, size=20)
    # Prove the softdtw branch forwards n_jobs/verbose to the metric
    captured = {}

    def spy_cdist_soft_dtw_normalized(dataset1, dataset2=None, **kwargs):
        captured.update(kwargs)
        dist = cdist_soft_dtw(dataset1, dataset2,
                              gamma=kwargs.get("gamma", 1.0))
        # Mirror _cdist_soft_dtw_normalized: subtract half the self-distances
        d_ii = np.diag(dist)
        dist = dist - 0.5 * (d_ii[:, None] + d_ii[None, :])
        np.fill_diagonal(dist, 0.)
        return dist

    monkeypatch.setattr(
        "tslearn.clustering.utils.cdist_soft_dtw_normalized",
        spy_cdist_soft_dtw_normalized)
    silhouette_samples(X, labels, metric="softdtw", n_jobs=2, verbose=1)
    assert captured.get("n_jobs") == 2
    assert captured.get("verbose") == 1
    # metric_params verbose is stripped so it cannot collide with the
    # explicit verbose argument
    silhouette_samples(X, labels, metric="softdtw",
                       metric_params={"verbose": 1})


@pytest.mark.parametrize("silhouette_func",
                         [silhouette_score, silhouette_samples])
@pytest.mark.parametrize(
    "metric_params, extra_kwds",
    [
        ({"be": "ignored", "n_jobs": -1, "verbose": 99}, {}),
        (None, {"be": "ignored"}),
    ],
    ids=["metric-params", "direct-keyword"]
)
def test_silhouette_controlled_metric_params_are_not_forwarded(
        monkeypatch, silhouette_func, metric_params, extra_kwds):
    X = np.arange(8, dtype=float).reshape(4, 2, 1)
    labels = np.array([0, 0, 1, 1])
    distances = np.array([
        [0., 1., 5., 6.],
        [1., 0., 4., 5.],
        [5., 4., 0., 1.],
        [6., 5., 1., 0.],
    ])
    captured = {}

    def fake_cdist_dtw(dataset1, dataset2=None, *, n_jobs=None, verbose=0,
                       be=None, **forwarded_params):
        captured.update(n_jobs=n_jobs, verbose=verbose, be=be,
                        forwarded_params=forwarded_params)
        return distances

    monkeypatch.setattr("tslearn.clustering.utils._cdist_dtw",
                        fake_cdist_dtw)

    silhouette_func(X, labels, metric="dtw", metric_params=metric_params,
                    n_jobs=2, verbose=3, **extra_kwds)

    assert captured["n_jobs"] == 2
    assert captured["verbose"] == 3
    assert captured["be"] != "ignored"
    assert captured["forwarded_params"] == {}


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

    # Softdtw-normalized with gamma metric param
    db.set_params(metric='softdtw_normalized')
    db.fit(X)
    np.testing.assert_equal(db.labels_, [-1, -1, -1, -1, -1, -1])
    db.set_params(eps=0.1)
    db.set_params(metric_params={'gamma': 0.1})
    db.fit(X)
    np.testing.assert_equal(db.labels_, [0, 0, 0, 1, 1, 1])
    np.testing.assert_equal(db.core_ts_indices_, [1, 4])

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
    db = TimeSeriesDBSCAN(n_jobs=5, metric_params={'n_jobs': 1}).fit(X)
    assert db._get_metric_params() == {'n_jobs': 5}

    # Ensure unused params don't raise
    TimeSeriesDBSCAN(metric="softdtw_normalized", n_jobs=1, metric_params={'n_jobs': 1}).fit(X)
    TimeSeriesDBSCAN(metric="dtw", metric_params={'gamma': 2}).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        TimeSeriesDBSCAN(metric="euclidean", metric_params={'whatever': "trimmed"}).fit(X)
