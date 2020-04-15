import numpy as np
from scipy.spatial.distance import cdist
import tslearn.metrics
import tslearn.clustering
from tslearn.utils import to_time_series

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_dtw():
    # dtw_path
    path, dist = tslearn.metrics.dtw_path([1, 2, 3], [1., 2., 2., 3.])
    np.testing.assert_equal(path, [(0, 0), (1, 1), (1, 2), (2, 3)])
    np.testing.assert_allclose(dist, 0.)

    path, dist = tslearn.metrics.dtw_path([1, 2, 3], [1., 2., 2., 3., 4.])
    np.testing.assert_allclose(dist, 1.)

    # dtw
    n1, n2, d = 15, 10, 3
    rng = np.random.RandomState(0)
    x = rng.randn(n1, d)
    y = rng.randn(n2, d)
    np.testing.assert_allclose(tslearn.metrics.dtw(x, y),
                               tslearn.metrics.dtw_path(x, y)[1])

    # cdist_dtw
    dists = tslearn.metrics.cdist_dtw([[1, 2, 2, 3],
                                       [1., 2., 3., 4.]])
    np.testing.assert_allclose(dists,
                               np.array([[0., 1.],
                                         [1., 0.]]))
    dists = tslearn.metrics.cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]],
                                      [[1, 2, 3], [2, 3, 4, 5]])
    np.testing.assert_allclose(dists,
                               np.array([[0., 2.44949],
                                         [1., 1.414214]]),
                               atol=1e-5)


def test_ldtw():
    n1, n2, d = 15, 10, 3
    rng = np.random.RandomState(0)
    x = rng.randn(n1, d)
    y = rng.randn(n2, d)

    # LDTW >= DTW
    assert tslearn.metrics.dtw(x, y) <= \
           tslearn.metrics.dtw_limited_warping_length(x, y, n1 + 2)

    # if path is too short, LDTW raises a ValueError
    np.testing.assert_raises(ValueError,
                             tslearn.metrics.dtw_limited_warping_length,
                             x, y, max(n1, n2) - 1)

    # if max_length is smaller than length of optimal DTW path, LDTW > DTW
    path, cost = tslearn.metrics.dtw_path(x, y)
    np.testing.assert_array_less(
        cost,
        tslearn.metrics.dtw_limited_warping_length(x, y, len(path) - 1)
    )

    # if max_length is geq than length of optimal DTW path, LDTW = DTW
    np.testing.assert_allclose(
        cost,
        tslearn.metrics.dtw_limited_warping_length(x, y, len(path))
    )
    np.testing.assert_allclose(
        cost,
        tslearn.metrics.dtw_limited_warping_length(x, y, len(path) + 1)
    )

    # test dtw_path_limited_warping_length
    path, cost = tslearn.metrics.dtw_path_limited_warping_length(x, y, n1 + 2)
    np.testing.assert_allclose(
        cost,
        tslearn.metrics.dtw_limited_warping_length(x, y, n1 + 2)
    )
    assert len(path) <= n1 + 2


def test_constrained_paths():
    n, d = 10, 3
    rng = np.random.RandomState(0)
    x = rng.randn(n, d)
    y = rng.randn(n, d)
    dtw_sakoe = tslearn.metrics.dtw(x, y,
                                    global_constraint="sakoe_chiba",
                                    sakoe_chiba_radius=0)
    dtw_itak = tslearn.metrics.dtw(x, y,
                                   global_constraint="itakura",
                                   itakura_max_slope=1.0)
    euc_dist = np.linalg.norm(x - y)
    np.testing.assert_allclose(dtw_sakoe, euc_dist,
                               atol=1e-5)
    np.testing.assert_allclose(dtw_itak, euc_dist,
                               atol=1e-5)

    z = rng.randn(3 * n, d)
    np.testing.assert_warns(RuntimeWarning, tslearn.metrics.dtw,
                            x, z, global_constraint="itakura",
                            itakura_max_slope=2.0)
    np.testing.assert_warns(RuntimeWarning, tslearn.metrics.dtw,
                            z, x, global_constraint="itakura",
                            itakura_max_slope=2.0)



def test_dtw_subseq():
    path, dist = tslearn.metrics.dtw_subsequence_path([2, 3],
                                                      [1., 2., 2., 3., 4.])
    np.testing.assert_equal(path, [(0, 2), (1, 3)])
    np.testing.assert_allclose(dist, 0.)

    path, dist = tslearn.metrics.dtw_subsequence_path([1, 4],
                                                      [1., 2., 2., 3., 4.])
    np.testing.assert_equal(path, [(0, 2), (1, 3)])
    np.testing.assert_allclose(dist, np.sqrt(2.))


def test_dtw_subseq_path():
    subseq, longseq = [1, 4], [1., 2., 2., 3., 4.]
    subseq = to_time_series(subseq)
    longseq = to_time_series(longseq)
    cost_matrix = tslearn.metrics.subsequence_cost_matrix(subseq, longseq)

    path = tslearn.metrics.subsequence_path(cost_matrix, 3)
    np.testing.assert_equal(path, [(0, 2), (1, 3)])

    path = tslearn.metrics.subsequence_path(cost_matrix, 1)
    np.testing.assert_equal(path, [(0, 0), (1, 1)])


def test_masks():
    sk_mask = tslearn.metrics.sakoe_chiba_mask(4, 4, 1)
    reference_mask = np.array([[0., 0., np.inf, np.inf],
                               [0., 0., 0., np.inf],
                               [np.inf, 0., 0., 0.],
                               [np.inf, np.inf, 0., 0.]])
    np.testing.assert_allclose(sk_mask, reference_mask)

    sk_mask = tslearn.metrics.sakoe_chiba_mask(7, 3, 1)
    reference_mask = np.array([[0., 0., np.inf],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [0., 0., 0.],
                               [np.inf, 0., 0.]])
    np.testing.assert_allclose(sk_mask, reference_mask)

    i_mask = tslearn.metrics.itakura_mask(6, 6)
    reference_mask = np.array([[0., np.inf, np.inf, np.inf, np.inf, np.inf],
                               [np.inf, 0., 0., np.inf, np.inf, np.inf],
                               [np.inf, 0., 0., 0., np.inf, np.inf],
                               [np.inf, np.inf, 0., 0., 0., np.inf],
                               [np.inf, np.inf, np.inf, 0., 0., np.inf],
                               [np.inf, np.inf, np.inf, np.inf, np.inf,  0.]])
    np.testing.assert_allclose(i_mask, reference_mask)

    # Test masks for different combinations of global_constraints /
    # sakoe_chiba_radius / itakura_max_slope
    sz = 10
    ts0 = np.empty((sz, 1))
    ts1 = np.empty((sz, 1))
    mask_no_constraint = tslearn.metrics.compute_mask(ts0, ts1,
                                                      global_constraint=0)
    np.testing.assert_allclose(mask_no_constraint, np.zeros((sz, sz)))

    mask_itakura = tslearn.metrics.compute_mask(ts0, ts1,
                                                global_constraint=1)
    mask_itakura_bis = tslearn.metrics.compute_mask(ts0, ts1,
                                                    itakura_max_slope=2.)
    np.testing.assert_allclose(mask_itakura, mask_itakura_bis)

    mask_sakoe = tslearn.metrics.compute_mask(ts0, ts1,
                                              global_constraint=2)

    mask_sakoe_bis = tslearn.metrics.compute_mask(ts0, ts1,
                                                  sakoe_chiba_radius=1)
    np.testing.assert_allclose(mask_sakoe, mask_sakoe_bis)

    np.testing.assert_raises(RuntimeWarning,
                             tslearn.metrics.compute_mask,
                             ts0, ts1,
                             sakoe_chiba_radius=1,
                             itakura_max_slope=2.)

    # Tests for estimators that can set masks through metric_params
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    estimator1 = tslearn.clustering.TimeSeriesKMeans(
        metric="dtw",
        metric_params={
            "itakura_max_slope": 1.0
        },
        max_iter=5,
        random_state=0
    )
    estimator2 = tslearn.clustering.TimeSeriesKMeans(metric="euclidean",
                                                     max_iter=5,
                                                     random_state=0)
    estimator3 = tslearn.clustering.TimeSeriesKMeans(
        metric="dtw",
        metric_params={
            "sakoe_chiba_radius": 0
        },
        max_iter=5,
        random_state=0
    )
    np.testing.assert_allclose(estimator1.fit(time_series).labels_,
                               estimator2.fit(time_series).labels_)
    np.testing.assert_allclose(estimator1.fit(time_series).labels_,
                               estimator3.fit(time_series).labels_)


def test_gak():
    # GAK
    g = tslearn.metrics.cdist_gak([[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)
    np.testing.assert_allclose(g,
                               np.array([[1., 0.656297],
                                         [0.656297, 1.]]),
                               atol=1e-5)
    g = tslearn.metrics.cdist_gak([[1, 2, 2], [1., 2., 3., 4.]],
                                  [[1, 2, 2, 3], [1., 2., 3., 4.]], sigma=2.)
    np.testing.assert_allclose(g,
                               np.array([[0.710595, 0.297229],
                                         [0.656297, 1.]]),
                               atol=1e-5)

    # soft-DTW
    d = tslearn.metrics.cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]],
                                       gamma=.01)
    np.testing.assert_allclose(d,
                               np.array([[-0.010986, 1.],
                                         [1., 0.]]),
                               atol=1e-5)
    d = tslearn.metrics.cdist_soft_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]],
                                       [[1, 2, 2, 3], [1., 2., 3., 4.]],
                                       gamma=.01)
    np.testing.assert_allclose(d,
                               np.array([[-0.010986, 1.],
                                         [1., 0.]]),
                               atol=1e-5)

    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    dists = tslearn.metrics.cdist_soft_dtw_normalized(time_series)
    np.testing.assert_equal(dists >= 0., True)

    v1 = rng.randn(n, 1)
    v2 = rng.randn(n, 1)
    sqeuc = tslearn.metrics.SquaredEuclidean(v1.flat, v2.flat)
    np.testing.assert_allclose(sqeuc.compute(),
                               cdist(v1, v2, metric="sqeuclidean"))


def test_symmetric_cdist():
    rng = np.random.RandomState(0)
    dataset = rng.randn(5, 10, 2)
    np.testing.assert_allclose(tslearn.metrics.cdist_dtw(dataset, dataset),
                               tslearn.metrics.cdist_dtw(dataset))
    np.testing.assert_allclose(tslearn.metrics.cdist_gak(dataset, dataset),
                               tslearn.metrics.cdist_gak(dataset))
    np.testing.assert_allclose(
        tslearn.metrics.cdist_soft_dtw(dataset, dataset),
        tslearn.metrics.cdist_soft_dtw(dataset))
    np.testing.assert_allclose(
        tslearn.metrics.cdist_soft_dtw_normalized(dataset, dataset),
        tslearn.metrics.cdist_soft_dtw_normalized(dataset))


def test_lb_keogh():
    ts1 = [1, 2, 3, 2, 1]
    env_low, env_up = tslearn.metrics.lb_envelope(ts1, radius=1)
    np.testing.assert_allclose(env_low,
                               np.array([[1.], [1.], [2.], [1.], [1.]]))
    np.testing.assert_allclose(env_up,
                               np.array([[2.], [3.], [3.], [3.], [2.]]))
