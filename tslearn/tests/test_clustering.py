import numpy as np

from tslearn.utils import to_time_series_dataset
from tslearn.clustering import EmptyClusterError, _check_full_length, \
    _check_no_empty_cluster, TimeSeriesKMeans,  GlobalAlignmentKernelKMeans, \
    KShape
from tslearn.metrics import cdist_dtw, cdist_soft_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from scipy.spatial.distance import cdist

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_check_no_empty_cluster():
    labels = np.array([1, 1, 2, 0, 2])
    _check_no_empty_cluster(labels, 3)
    np.testing.assert_raises(EmptyClusterError, _check_no_empty_cluster,
                             labels, 4)


def test_check_full_length():
    centroids = to_time_series_dataset([[1, 2, 3], [1, 2, 3, 4, 5]])
    arr = _check_full_length(centroids)
    np.testing.assert_allclose(arr, to_time_series_dataset([[1, 2, 3, 3, 3],
                                                            [1, 2, 3, 4, 5]]))


def test_gak_kmeans():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)

    gak_km = GlobalAlignmentKernelKMeans(n_clusters=3, verbose=False,
                                         max_iter=5,
                                         random_state=rng).fit(time_series)
    np.testing.assert_allclose(gak_km.labels_, gak_km.predict(time_series))

    gak_km = GlobalAlignmentKernelKMeans(n_clusters=101, verbose=False,
                                         max_iter=5,
                                         random_state=rng).fit(time_series)
    assert gak_km._X_fit is None


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

    assert KShape(n_clusters=101, verbose=False,
                  random_state=rng).fit(time_series)._X_fit is None
