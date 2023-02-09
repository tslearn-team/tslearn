import numpy as np
import pytest
from scipy.spatial.distance import cdist

import tslearn.clustering
import tslearn.metrics
from tslearn.backend.backend import Backend
from tslearn.metrics.dtw_variants import dtw_path
from tslearn.utils import to_time_series

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


def test_dtw():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        # dtw_path
        path, dist = tslearn.metrics.dtw_path([1, 2, 3], [1.0, 2.0, 2.0, 3.0], be=be)
        np.testing.assert_equal(path, [(0, 0), (1, 1), (1, 2), (2, 3)])
        np.testing.assert_allclose(dist, be.array([0.0]))

        path, dist = tslearn.metrics.dtw_path(
            [1, 2, 3], [1.0, 2.0, 2.0, 3.0, 4.0], be=be
        )
        np.testing.assert_allclose(dist, be.array([1.0]))

        # dtw
        n1, n2, d = 15, 10, 3
        rng = np.random.RandomState(0)
        x = be.array(rng.randn(n1, d))
        y = be.array(rng.randn(n2, d))

        np.testing.assert_allclose(
            tslearn.metrics.dtw(x, y, be=be), tslearn.metrics.dtw_path(x, y, be=be)[1]
        )

        # cdist_dtw
        dists = tslearn.metrics.cdist_dtw([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], be=be)
        np.testing.assert_allclose(dists, be.array([[0.0, 1.0], [1.0, 0.0]]))
        dists = tslearn.metrics.cdist_dtw(
            [[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], [[1, 2, 3], [2, 3, 4, 5]], be=be
        )
        np.testing.assert_allclose(
            dists, be.array([[0.0, 2.44949], [1.0, 1.414214]]), atol=1e-5
        )


def test_ctw():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        # ctw_path
        path, cca, dist = tslearn.metrics.ctw_path(
            [1, 2, 3], [1.0, 2.0, 2.0, 3.0], be=be
        )
        np.testing.assert_equal(path, [(0, 0), (1, 1), (1, 2), (2, 3)])
        np.testing.assert_allclose(dist, 0.0)

        path, cca, dist = tslearn.metrics.ctw_path(
            [1, 2, 3], [1.0, 2.0, 2.0, 3.0, 4.0], be=be
        )
        np.testing.assert_allclose(dist, 1.0)

        # dtw
        n1, n2, d1, d2 = 15, 10, 3, 1
        rng = np.random.RandomState(0)
        x = rng.randn(n1, d1)
        y = rng.randn(n2, d2)
        x, y = be.array(x), be.array(y)
        np.testing.assert_allclose(
            tslearn.metrics.ctw(x, y, be=be), tslearn.metrics.ctw(y, x, be=be)
        )
        np.testing.assert_allclose(
            tslearn.metrics.ctw(x, y, be=be), tslearn.metrics.ctw_path(x, y, be=be)[-1]
        )

        # cdist_dtw
        dists = tslearn.metrics.cdist_ctw([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], be=be)
        np.testing.assert_allclose(dists, be.array([[0.0, 1.0], [1.0, 0.0]]))
        dists = tslearn.metrics.cdist_ctw(
            [[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], [[1, 2, 3], [2, 3, 4, 5]], be=be
        )
        np.testing.assert_allclose(
            dists, be.array([[0.0, 2.44949], [1.0, 1.414214]]), atol=1e-5
        )


def test_ldtw():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)

        n1, n2, d = 15, 10, 3
        rng = np.random.RandomState(0)
        x = be.array(rng.randn(n1, d))
        y = be.array(rng.randn(n2, d))

        # LDTW >= DTW
        np.testing.assert_allclose(
            tslearn.metrics.dtw(x, y, be=be),
            tslearn.metrics.dtw_limited_warping_length(x, y, n1 + 2, be=be),
        )

        # if path is too short, LDTW raises a ValueError
        np.testing.assert_raises(
            ValueError,
            tslearn.metrics.dtw_limited_warping_length,
            x,
            y,
            max(n1, n2) - 1,
            be,
        )

        # if max_length is smaller than length of optimal DTW path, LDTW > DTW
        path, cost = tslearn.metrics.dtw_path(x, y, be=be)
        np.testing.assert_array_less(
            cost, tslearn.metrics.dtw_limited_warping_length(x, y, len(path) - 1, be=be)
        )

        # if max_length is geq than length of optimal DTW path, LDTW = DTW
        np.testing.assert_allclose(
            cost, tslearn.metrics.dtw_limited_warping_length(x, y, len(path))
        )
        np.testing.assert_allclose(
            cost, tslearn.metrics.dtw_limited_warping_length(x, y, len(path) + 1, be=be)
        )
        path, cost = tslearn.metrics.dtw_path_limited_warping_length(
            x, y, n1 + 2, be=be
        )
        np.testing.assert_allclose(
            cost, tslearn.metrics.dtw_limited_warping_length(x, y, n1 + 2, be=be)
        )
        assert len(path) <= n1 + 2


def test_lcss():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        sim = tslearn.metrics.lcss([1, 2, 3], [1.0, 2.0, 2.0, 3.0], be=be)
        np.testing.assert_equal(sim, 1.0)

        sim = tslearn.metrics.lcss([1, 2, 3], [1.0, 2.0, 2.0, 4.0], be=be)
        np.testing.assert_equal(sim, 1.0)

        sim = tslearn.metrics.lcss([1, 2, 3], [-2.0, 5.0, 7.0], eps=3, be=be)
        np.testing.assert_equal(round(sim, 2), 0.67)

        sim = tslearn.metrics.lcss([1, 2, 3], [1.0, 2.0, 2.0, 2.0, 3.0], eps=0, be=be)
        np.testing.assert_equal(sim, 1.0)


def test_lcss_path():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        path, sim = tslearn.metrics.lcss_path(
            [1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0], be=be
        )
        np.testing.assert_equal(sim, 1.0)
        np.testing.assert_equal(path, [(0, 1), (1, 2), (2, 3)])

        path, sim = tslearn.metrics.lcss_path(
            [1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 4.0], be=be
        )
        np.testing.assert_equal(sim, 1.0)
        np.testing.assert_equal(path, [(0, 1), (1, 2), (2, 3)])

        path, sim = tslearn.metrics.lcss_path(
            [1.0, 2.0, 3.0], [-2.0, 5.0, 7.0], eps=3, be=be
        )
        np.testing.assert_equal(round(sim, 2), 0.67)
        np.testing.assert_equal(path, [(0, 0), (2, 1)])

        path, sim = tslearn.metrics.lcss_path(
            [1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 2.0, 3.0], eps=0, be=be
        )
        np.testing.assert_equal(sim, 1.0)
        np.testing.assert_equal(path, [(0, 0), (1, 3), (2, 4)])


def test_lcss_path_from_metric():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        for d in be.arange(1, 5):
            rng = np.random.RandomState()
            s1, s2 = rng.randn(10, d), rng.randn(30, d)
            s1, s2 = be.array(s1), be.array(s2)

            # Use lcss_path as a reference
            path_ref, sim_ref = tslearn.metrics.lcss_path(s1, s2, be=be)

            # Test of using a scipy distance function
            path, sim = tslearn.metrics.lcss_path_from_metric(
                s1, s2, metric="sqeuclidean", be=be
            )

            np.testing.assert_equal(path, path_ref)
            np.testing.assert_equal(sim, sim_ref)

            # Test of defining a custom function
            def sqeuclidean(x, y):
                return be.sum((x - y) ** 2)

            path, sim = tslearn.metrics.lcss_path_from_metric(
                s1, s2, metric=sqeuclidean, be=be
            )
            np.testing.assert_equal(path, path_ref)
            np.testing.assert_equal(sim, sim_ref)

            # Test of precomputing the distance matrix
            dist_matrix = cdist(s1, s2, metric="sqeuclidean")
            path, sim = tslearn.metrics.lcss_path_from_metric(
                dist_matrix, metric="precomputed", be=be
            )
            np.testing.assert_equal(path, path_ref)
            np.testing.assert_equal(sim, sim_ref)


def test_constrained_paths():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        n, d = 10, 3
        rng = np.random.RandomState(0)
        x = rng.randn(n, d)
        y = rng.randn(n, d)
        x, y = be.array(x), be.array(y)
        dtw_sakoe = tslearn.metrics.dtw(
            x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=0, be=be
        )
        dtw_itak = tslearn.metrics.dtw(
            x, y, global_constraint="itakura", itakura_max_slope=1.0, be=be
        )
        euc_dist = be.linalg.norm(x - y)
        np.testing.assert_allclose(dtw_sakoe, euc_dist, atol=1e-5)
        np.testing.assert_allclose(dtw_itak, euc_dist, atol=1e-5)

        z = rng.randn(3 * n, d)
        z = be.array(z)
        np.testing.assert_warns(
            RuntimeWarning,
            tslearn.metrics.dtw,
            x,
            z,
            global_constraint="itakura",
            itakura_max_slope=2.0,
            be=be,
        )
        np.testing.assert_warns(
            RuntimeWarning,
            tslearn.metrics.dtw,
            z,
            x,
            global_constraint="itakura",
            itakura_max_slope=2.0,
            be=be,
        )


def test_dtw_subseq():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)

        path, dist = tslearn.metrics.dtw_subsequence_path(
            [2, 3], [1.0, 2.0, 2.0, 3.0, 4.0], be=be
        )
        np.testing.assert_allclose(path, [(0, 2), (1, 3)])
        np.testing.assert_allclose(dist, 0.0)

        path, dist = tslearn.metrics.dtw_subsequence_path(
            [1, 4], [1.0, 2.0, 2.0, 3.0, 4.0], be=be
        )
        np.testing.assert_allclose(path, [(0, 2), (1, 3)])
        np.testing.assert_allclose(dist, np.sqrt(2.0))


def test_dtw_subseq_path():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        subseq, longseq = [1, 4], [1.0, 2.0, 2.0, 3.0, 4.0]
        subseq = to_time_series(subseq, be=be)
        longseq = to_time_series(longseq, be=be)
        cost_matrix = tslearn.metrics.subsequence_cost_matrix(subseq, longseq, be=be)

        path = tslearn.metrics.subsequence_path(cost_matrix, 3, be=be)
        np.testing.assert_equal(path, [(0, 2), (1, 3)])

        path = tslearn.metrics.subsequence_path(cost_matrix, 1, be=be)
        np.testing.assert_equal(path, [(0, 0), (1, 1)])


def test_masks():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        sk_mask = tslearn.metrics.sakoe_chiba_mask(4, 4, 1, be=be)
        reference_mask = be.array(
            [
                [0.0, 0.0, be.inf, be.inf],
                [0.0, 0.0, 0.0, be.inf],
                [be.inf, 0.0, 0.0, 0.0],
                [be.inf, be.inf, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(sk_mask, reference_mask)

        sk_mask = tslearn.metrics.sakoe_chiba_mask(7, 3, 1, be=be)
        reference_mask = be.array(
            [
                [0.0, 0.0, be.inf],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [be.inf, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(sk_mask, reference_mask)

        i_mask = tslearn.metrics.itakura_mask(6, 6, be=be)
        reference_mask = be.array(
            [
                [0.0, be.inf, be.inf, be.inf, be.inf, be.inf],
                [be.inf, 0.0, 0.0, be.inf, be.inf, be.inf],
                [be.inf, 0.0, 0.0, 0.0, be.inf, be.inf],
                [be.inf, be.inf, 0.0, 0.0, 0.0, be.inf],
                [be.inf, be.inf, be.inf, 0.0, 0.0, be.inf],
                [be.inf, be.inf, be.inf, be.inf, be.inf, 0.0],
            ]
        )
        np.testing.assert_allclose(i_mask, reference_mask)

        # Test masks for different combinations of global_constraints /
        # sakoe_chiba_radius / itakura_max_slope
        sz = 10
        ts0 = be.empty((sz, 1))
        ts1 = be.empty((sz, 1))
        mask_no_constraint = tslearn.metrics.dtw_variants.compute_mask(
            ts0, ts1, global_constraint=0, be=be
        )
        np.testing.assert_allclose(mask_no_constraint, be.zeros((sz, sz)))

        mask_itakura = tslearn.metrics.dtw_variants.compute_mask(
            ts0, ts1, global_constraint=1, be=be
        )
        mask_itakura_bis = tslearn.metrics.dtw_variants.compute_mask(
            ts0, ts1, itakura_max_slope=2.0, be=be
        )
        np.testing.assert_allclose(mask_itakura, mask_itakura_bis)

        mask_sakoe = tslearn.metrics.dtw_variants.compute_mask(
            ts0, ts1, global_constraint=2, be=be
        )

        mask_sakoe_bis = tslearn.metrics.dtw_variants.compute_mask(
            ts0, ts1, sakoe_chiba_radius=1, be=be
        )
        np.testing.assert_allclose(mask_sakoe, mask_sakoe_bis)

        np.testing.assert_raises(
            RuntimeWarning,
            tslearn.metrics.dtw_variants.compute_mask,
            ts0,
            ts1,
            sakoe_chiba_radius=1,
            itakura_max_slope=2.0,
            be=be,
        )

        # Tests for estimators that can set masks through metric_params
        n, sz, d = 15, 10, 3
        rng = np.random.RandomState(0)
        time_series = rng.randn(n, sz, d)
        time_series = be.array(time_series)
        estimator1 = tslearn.clustering.TimeSeriesKMeans(
            metric="dtw",
            metric_params={"itakura_max_slope": 1.0},
            max_iter=5,
            random_state=0,
        )
        estimator2 = tslearn.clustering.TimeSeriesKMeans(
            metric="euclidean", max_iter=5, random_state=0
        )
        estimator3 = tslearn.clustering.TimeSeriesKMeans(
            metric="dtw",
            metric_params={"sakoe_chiba_radius": 0},
            max_iter=5,
            random_state=0,
        )
        np.testing.assert_allclose(
            estimator1.fit(time_series).labels_, estimator2.fit(time_series).labels_
        )
        np.testing.assert_allclose(
            estimator1.fit(time_series).labels_, estimator3.fit(time_series).labels_
        )


def test_gak():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        # GAK
        g = tslearn.metrics.cdist_gak(
            [[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], sigma=2.0, be=be
        )
        np.testing.assert_allclose(
            g, be.array([[1.0, 0.656297], [0.656297, 1.0]]), atol=1e-5
        )
        g = tslearn.metrics.cdist_gak(
            [[1, 2, 2], [1.0, 2.0, 3.0, 4.0]],
            [[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]],
            sigma=2.0,
            be=be,
        )
        np.testing.assert_allclose(
            g, be.array([[0.710595, 0.297229], [0.656297, 1.0]]), atol=1e-5
        )

        # soft-DTW
        d = tslearn.metrics.cdist_soft_dtw(
            [[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], gamma=0.01, be=be
        )
        np.testing.assert_allclose(
            d, be.array([[-0.010986, 1.0], [1.0, 0.0]]), atol=1e-5
        )
        d = tslearn.metrics.cdist_soft_dtw(
            [[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]],
            [[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]],
            gamma=0.01,
            be=be,
        )
        np.testing.assert_allclose(
            d, be.array([[-0.010986, 1.0], [1.0, 0.0]]), atol=1e-5
        )

        n, sz, d = 15, 10, 3
        rng = np.random.RandomState(0)
        time_series = rng.randn(n, sz, d)
        time_series = be.array(time_series)
        dists = tslearn.metrics.cdist_soft_dtw_normalized(time_series, be=be)
        assert be.all(dists >= 0)

        v1 = rng.randn(n, 1)
        v2 = rng.randn(n, 1)
        sqeuc = tslearn.metrics.SquaredEuclidean(v1.flat, v2.flat, be=be)
        np.testing.assert_allclose(sqeuc.compute(), cdist(v1, v2, metric="sqeuclidean"))


def test_gamma_soft_dtw():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        dataset = be.array([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]])
        gamma = tslearn.metrics.gamma_soft_dtw(
            dataset=dataset, n_samples=200, random_state=0, be=be
        )
        np.testing.assert_allclose(gamma, 8.0)


def test_symmetric_cdist():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        rng = np.random.RandomState(0)
        dataset = rng.randn(5, 10, 2)
        dataset = be.array(dataset)
        np.testing.assert_allclose(
            tslearn.metrics.cdist_dtw(dataset, dataset, be=be),
            tslearn.metrics.cdist_dtw(dataset, be=be),
        )
        np.testing.assert_allclose(
            tslearn.metrics.cdist_gak(dataset, dataset, be=be),
            tslearn.metrics.cdist_gak(dataset, be=be),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            tslearn.metrics.cdist_soft_dtw(dataset, dataset, be=be),
            tslearn.metrics.cdist_soft_dtw(dataset, be=be),
        )
        np.testing.assert_allclose(
            tslearn.metrics.cdist_soft_dtw_normalized(dataset, dataset, be=be),
            tslearn.metrics.cdist_soft_dtw_normalized(dataset, be=be),
        )


def test_lb_keogh():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        ts1 = [1, 2, 3, 2, 1]
        env_low, env_up = tslearn.metrics.lb_envelope(ts1, radius=1, be=be)
        np.testing.assert_allclose(
            env_low, be.array([[1.0], [1.0], [2.0], [1.0], [1.0]])
        )
        np.testing.assert_allclose(
            env_up, be.array([[2.0], [3.0], [3.0], [3.0], [2.0]])
        )


def test_dtw_path_from_metric():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        rng = np.random.RandomState(0)
        s1, s2 = rng.rand(10, 2), rng.rand(30, 2)
        s1, s2 = be.array(s1), be.array(s2)

        # Use dtw_path as a reference
        path_ref, dist_ref = tslearn.metrics.dtw_path(s1, s2, be=be)

        # Test of using a scipy distance function
        path, dist = tslearn.metrics.dtw_path_from_metric(
            s1, s2, metric="sqeuclidean", be=be
        )
        np.testing.assert_equal(path, path_ref)
        np.testing.assert_allclose(be.sqrt(dist), dist_ref)

        # Test of defining a custom function
        def sqeuclidean(x, y):
            return be.sum((x - y) ** 2)

        path, dist = tslearn.metrics.dtw_path_from_metric(
            s1, s2, metric=sqeuclidean, be=be
        )
        np.testing.assert_equal(path, path_ref)
        np.testing.assert_allclose(be.sqrt(dist), dist_ref)

        # Test of precomputing the distance matrix
        dist_matrix = cdist(s1, s2, metric="sqeuclidean")
        dist_matrix = be.array(dist_matrix)
        path, dist = tslearn.metrics.dtw_path_from_metric(
            dist_matrix, metric="precomputed", be=be
        )
        np.testing.assert_equal(path, path_ref)
        np.testing.assert_allclose(be.sqrt(dist), dist_ref)


def test_softdtw():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        rng = np.random.RandomState(0)
        s1, s2 = rng.rand(10, 2), rng.rand(30, 2)
        s1, s2 = be.array(s1), be.array(s2)

        # Use dtw_path as a reference
        path_ref, dist_ref = tslearn.metrics.dtw_path(s1, s2, be=be)
        mat_path_ref = be.zeros((10, 30))
        for i, j in path_ref:
            mat_path_ref[i, j] = 1.0

        # Test of using a scipy distance function
        matrix_path, dist = tslearn.metrics.soft_dtw_alignment(s1, s2, gamma=0.0, be=be)

        np.testing.assert_equal(dist, dist_ref**2)
        np.testing.assert_allclose(matrix_path, mat_path_ref)


def test_dtw_path_with_empty_or_nan_inputs():
    BACKENDS = ["numpy", "pytorch"]
    for backend in BACKENDS:
        be = Backend(backend)
        s1 = be.zeros((3, 10))
        s2_empty = be.zeros((0, 10))
        with pytest.raises(ValueError) as excinfo:
            dtw_path(s1, s2_empty, be=be)
        assert (
            str(excinfo.value)
            == "One of the input time series contains only nans or has zero length."
        )

        s2_nan = be.full((3, 10), be.nan)
        with pytest.raises(ValueError) as excinfo:
            dtw_path(s1, s2_nan, be=be)
        assert (
            str(excinfo.value)
            == "One of the input time series contains only nans or has zero length."
        )
