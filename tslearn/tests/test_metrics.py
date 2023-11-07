import platform
import sys

import numpy as np
import pytest
import tslearn.clustering
import tslearn.metrics
from scipy.spatial.distance import cdist
from tslearn.backend.backend import Backend, cast, instantiate_backend
from tslearn.metrics.dtw_variants import dtw_path
from tslearn.utils import to_time_series

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"

try:
    import torch
    backends = [Backend("numpy"), Backend("pytorch"), None]
    array_types = ["numpy", "pytorch", "list"]
except ImportError:
    backends = [Backend("numpy")]
    array_types = ["numpy", "list"]


def test_dtw():
    for be in backends:
        for array_type in array_types:
            backend = instantiate_backend(be, array_type)
            # dtw_path
            path, dist = tslearn.metrics.dtw_path(cast([1, 2, 3], array_type), cast([1.0, 2.0, 2.0, 3.0], array_type), be=be)
            np.testing.assert_equal(path, [(0, 0), (1, 1), (1, 2), (2, 3)])
            np.testing.assert_allclose(dist, [0.0])
            assert backend.belongs_to_backend(dist)

            path, dist = tslearn.metrics.dtw_path(
                cast([1, 2, 3], array_type), cast([1.0, 2.0, 2.0, 3.0, 4.0], array_type), be=be
            )
            np.testing.assert_allclose(dist, [1.0])
            assert backend.belongs_to_backend(dist)

            # dtw
            n1, n2, d = 15, 10, 3
            rng = np.random.RandomState(0)
            x = cast(rng.randn(n1, d), array_type)
            y = cast(rng.randn(n2, d), array_type)

            np.testing.assert_allclose(
                tslearn.metrics.dtw(x, y, be=be), tslearn.metrics.dtw_path(x, y, be=be)[1]
            )

            # cdist_dtw
            dists = tslearn.metrics.cdist_dtw(cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type), be=be)
            np.testing.assert_allclose(dists, cast([[0.0, 1.0], [1.0, 0.0]], array_type))
            dists = tslearn.metrics.cdist_dtw(
                cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type),
                [[1, 2, 3], [2, 3, 4, 5]],  # The second dataset can not be cast to array because of its shape
                be=be
            )
            np.testing.assert_allclose(dists, [[0.0, 2.44949], [1.0, 1.414214]], atol=1e-5)
            assert backend.belongs_to_backend(dists)


def test_ctw():
    for be in backends:
        for array_type in array_types:
            backend = instantiate_backend(be, array_type)
            # ctw_path
            path, cca, dist = tslearn.metrics.ctw_path(
                cast([1, 2, 3], array_type), cast([1.0, 2.0, 2.0, 3.0], array_type), be=be
            )
            np.testing.assert_equal(path, [(0, 0), (1, 1), (1, 2), (2, 3)])
            np.testing.assert_allclose(dist, 0.0)
            assert backend.belongs_to_backend(dist)

            path, cca, dist = tslearn.metrics.ctw_path(
                cast([1, 2, 3], array_type), cast([1.0, 2.0, 2.0, 3.0, 4.0], array_type), be=be
            )
            np.testing.assert_allclose(dist, 1.0)
            assert backend.belongs_to_backend(dist)

            # dtw
            n1, n2, d1, d2 = 15, 10, 3, 1
            rng = np.random.RandomState(0)
            x = cast(rng.randn(n1, d1), array_type)
            y = cast(rng.randn(n2, d2), array_type)
            np.testing.assert_allclose(
                tslearn.metrics.ctw(x, y, be=be), tslearn.metrics.ctw(y, x, be=be)
            )
            np.testing.assert_allclose(
                tslearn.metrics.ctw(x, y, be=be), tslearn.metrics.ctw_path(x, y, be=be)[-1]
            )

            # cdist_dtw
            dists = tslearn.metrics.cdist_ctw(cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type), be=be)
            np.testing.assert_allclose(dists, [[0.0, 1.0], [1.0, 0.0]])
            assert backend.belongs_to_backend(dist)

            dists = tslearn.metrics.cdist_ctw(
                cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type),
                [[1, 2, 3], [2, 3, 4, 5]], be=be  # The second dataset can not be cast to array because of its shape
            )
            np.testing.assert_allclose(
                dists, [[0.0, 2.44949], [1.0, 1.414214]], atol=1e-5
            )
            assert backend.belongs_to_backend(dists)


def test_ldtw():
    for be in backends:
        for array_type in array_types:
            n1, n2, d = 15, 10, 3
            rng = np.random.RandomState(0)
            x = cast(rng.randn(n1, d), array_type)
            y = cast(rng.randn(n2, d), array_type)

            ldtw_n1_plus_2 = tslearn.metrics.dtw_limited_warping_length(x, y, n1 + 2, be=be)

            # LDTW >= DTW
            np.testing.assert_allclose(
                tslearn.metrics.dtw(x, y, be=be),
                ldtw_n1_plus_2,
            )
            backend = instantiate_backend(be, array_type)
            assert backend.belongs_to_backend(ldtw_n1_plus_2)

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
                cost, ldtw_n1_plus_2
            )
            assert len(path) <= n1 + 2


def test_lcss():
    for be in backends:
        for array_type in array_types:
            sim = tslearn.metrics.lcss(cast([1, 2, 3], array_type), cast([1.0, 2.0, 2.0, 3.0], array_type), be=be)
            np.testing.assert_equal(sim, 1.0)
            assert isinstance(sim, float)

            sim = tslearn.metrics.lcss([1, 2, 3], [1.0, 2.0, 2.0, 4.0], be=be)
            np.testing.assert_equal(sim, 1.0)

            sim = tslearn.metrics.lcss([1, 2, 3], [-2.0, 5.0, 7.0], eps=3, be=be)
            np.testing.assert_equal(round(sim, 2), 0.67)

            sim = tslearn.metrics.lcss([1, 2, 3], [1.0, 2.0, 2.0, 2.0, 3.0], eps=0, be=be)
            np.testing.assert_equal(sim, 1.0)

            sim = tslearn.metrics.lcss(
                [[1, 1], [2, 2], [3, 3]], [[1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0]], eps=0, be=be)
            np.testing.assert_equal(sim, 1.0)


def test_lcss_path():
    for be in backends:
        for array_type in array_types:
            path, sim = tslearn.metrics.lcss_path(
                cast([1.0, 2.0, 3.0], array_type), cast([1.0, 2.0, 2.0, 3.0], array_type), be=be
            )
            np.testing.assert_equal(sim, 1.0)
            assert isinstance(sim, float)
            np.testing.assert_equal(path, [(0, 1), (1, 2), (2, 3)])
            assert isinstance(path, list)

            path, sim = tslearn.metrics.lcss_path(
                cast([1.0, 2.0, 3.0], array_type), cast([1.0, 2.0, 2.0, 4.0], array_type), be=be
            )
            np.testing.assert_equal(sim, 1.0)
            np.testing.assert_equal(path, [(0, 1), (1, 2), (2, 3)])

            path, sim = tslearn.metrics.lcss_path(
                cast([1.0, 2.0, 3.0], array_type), cast([-2.0, 5.0, 7.0], array_type), eps=3, be=be
            )
            np.testing.assert_equal(round(sim, 2), 0.67)
            np.testing.assert_equal(path, [(0, 0), (2, 1)])

            path, sim = tslearn.metrics.lcss_path(
                cast([1.0, 2.0, 3.0], array_type), cast([1.0, 2.0, 2.0, 2.0, 3.0], array_type), eps=0, be=be
            )
            np.testing.assert_equal(sim, 1.0)
            np.testing.assert_equal(path, [(0, 0), (1, 3), (2, 4)])


def test_lcss_path_from_metric():
    for be in backends:
        for array_type in array_types:
            for d in np.arange(1, 5):
                rng = np.random.RandomState(0)
                s1 = cast(rng.randn(10, d), array_type)
                s2 = cast(rng.randn(30, d), array_type)

                # Use lcss_path as a reference
                path_ref, sim_ref = tslearn.metrics.lcss_path(s1, s2, be=be)

                # Test of using a scipy distance function
                path, sim = tslearn.metrics.lcss_path_from_metric(
                    s1, s2, metric="sqeuclidean", be=be
                )

                np.testing.assert_equal(path, path_ref)
                assert isinstance(path, list)
                np.testing.assert_equal(sim, sim_ref)
                assert isinstance(sim, float)

                # Test of defining a custom function
                def sqeuclidean(x, y):
                    return sum((x - y) ** 2)

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
    for be in backends:
        for array_type in array_types:
            n, d = 10, 3
            rng = np.random.RandomState(0)
            x = cast(rng.randn(n, d), array_type)
            y = cast(rng.randn(n, d), array_type)
            dtw_sakoe = tslearn.metrics.dtw(
                x, y, global_constraint="sakoe_chiba", sakoe_chiba_radius=0, be=be
            )
            dtw_itak = tslearn.metrics.dtw(
                x, y, global_constraint="itakura", itakura_max_slope=1.0, be=be
            )
            backend = instantiate_backend(be, array_type)
            euc_dist = backend.linalg.norm(backend.array(x) - backend.array(y))
            np.testing.assert_allclose(dtw_sakoe, euc_dist, atol=1e-5)
            np.testing.assert_allclose(dtw_itak, euc_dist, atol=1e-5)
            backend = instantiate_backend(be, array_type)
            assert backend.is_float(dtw_sakoe)
            assert backend.is_float(dtw_itak)

            z = cast(rng.randn(3 * n, d), array_type)
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
    for be in backends:
        for array_type in array_types:
            path, dist = tslearn.metrics.dtw_subsequence_path(
                cast([2, 3], array_type), cast([1.0, 2.0, 2.0, 3.0, 4.0], array_type), be=be
            )
            np.testing.assert_allclose(path, [(0, 2), (1, 3)])
            np.testing.assert_allclose(dist, 0.0)
            backend = instantiate_backend(be, array_type)
            backend.belongs_to_backend(dist)

            path, dist = tslearn.metrics.dtw_subsequence_path(
                cast([1, 4], array_type), cast([1.0, 2.0, 2.0, 3.0, 4.0], array_type), be=be
            )
            np.testing.assert_allclose(path, [(0, 2), (1, 3)])
            np.testing.assert_allclose(dist, np.sqrt(2.0))
            assert backend.belongs_to_backend(dist)


def test_dtw_subseq_path():
    for be in backends:
        for array_type in array_types:
            backend = instantiate_backend(be, array_type)
            # subseq, longseq = [1, 4], [1.0, 2.0, 2.0, 3.0, 4.0]
            subseq = cast([1, 4], array_type)
            longseq = cast([1.0, 2.0, 2.0, 3.0, 4.0], array_type)
            subseq = to_time_series(subseq, be=None)
            longseq = to_time_series(longseq, be=None)
            cost_matrix = tslearn.metrics.subsequence_cost_matrix(subseq, longseq, be=be)
            assert backend.belongs_to_backend(cost_matrix)

            path = tslearn.metrics.subsequence_path(cost_matrix, 3, be=be)
            np.testing.assert_equal(path, [(0, 2), (1, 3)])
            assert isinstance(path, list)

            path = tslearn.metrics.subsequence_path(cost_matrix, 1, be=be)
            np.testing.assert_equal(path, [(0, 0), (1, 1)])
            assert isinstance(path, list)


def test_masks():
    for be in backends:
        for array_type in array_types:
            backend = instantiate_backend(be)
            sk_mask = tslearn.metrics.sakoe_chiba_mask(4, 4, 1, be=be)
            reference_mask = np.array(
                [
                    [0.0, 0.0, np.inf, np.inf],
                    [0.0, 0.0, 0.0, np.inf],
                    [np.inf, 0.0, 0.0, 0.0],
                    [np.inf, np.inf, 0.0, 0.0],
                ]
            )
            np.testing.assert_allclose(sk_mask, reference_mask)
            assert backend.belongs_to_backend(sk_mask)

            sk_mask = tslearn.metrics.sakoe_chiba_mask(7, 3, 1, be=be)
            reference_mask = np.array(
                [
                    [0.0, 0.0, np.inf],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [np.inf, 0.0, 0.0],
                ]
            )
            np.testing.assert_allclose(sk_mask, reference_mask)
            assert backend.belongs_to_backend(sk_mask)

            i_mask = tslearn.metrics.itakura_mask(6, 6, be=be)
            reference_mask = np.array(
                [
                    [0.0, np.inf, np.inf, np.inf, np.inf, np.inf],
                    [np.inf, 0.0, 0.0, np.inf, np.inf, np.inf],
                    [np.inf, 0.0, 0.0, 0.0, np.inf, np.inf],
                    [np.inf, np.inf, 0.0, 0.0, 0.0, np.inf],
                    [np.inf, np.inf, np.inf, 0.0, 0.0, np.inf],
                    [np.inf, np.inf, np.inf, np.inf, np.inf, 0.0],
                ]
            )
            np.testing.assert_allclose(i_mask, reference_mask)
            assert backend.belongs_to_backend(i_mask)

            # Test masks for different combinations of global_constraints /
            # sakoe_chiba_radius / itakura_max_slope
            sz = 10
            ts0 = cast(np.empty((sz, 1)), array_type)
            ts1 = cast(np.empty((sz, 1)), array_type)
            mask_no_constraint = tslearn.metrics.dtw_variants.compute_mask(
                ts0, ts1, global_constraint=0, be=be
            )
            np.testing.assert_allclose(mask_no_constraint, np.zeros((sz, sz)))
            backend = instantiate_backend(be, array_type)
            assert backend.belongs_to_backend(mask_no_constraint)

            mask_itakura = tslearn.metrics.dtw_variants.compute_mask(
                ts0, ts1, global_constraint=1, be=be
            )
            mask_itakura_bis = tslearn.metrics.dtw_variants.compute_mask(
                ts0, ts1, itakura_max_slope=2.0, be=be
            )
            np.testing.assert_allclose(mask_itakura, mask_itakura_bis)
            assert backend.belongs_to_backend(mask_itakura)

            mask_sakoe = tslearn.metrics.dtw_variants.compute_mask(
                ts0, ts1, global_constraint=2, be=be
            )

            mask_sakoe_bis = tslearn.metrics.dtw_variants.compute_mask(
                ts0, ts1, sakoe_chiba_radius=1, be=be
            )
            np.testing.assert_allclose(mask_sakoe, mask_sakoe_bis)
            assert backend.belongs_to_backend(mask_sakoe)

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
            time_series = cast(time_series, array_type)
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
    for be in backends:
        for array_type in array_types:
            backend = instantiate_backend(be, array_type)
            # GAK
            g = tslearn.metrics.cdist_gak(
                cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type), sigma=2.0, be=be
            )
            np.testing.assert_allclose(
                g, np.array([[1.0, 0.656297], [0.656297, 1.0]]), atol=1e-5
            )
            assert backend.belongs_to_backend(g)
            g = tslearn.metrics.cdist_gak(
                [[1, 2, 2], [1.0, 2.0, 3.0, 4.0]],  # Can not be cast to array because of its shape
                cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type),
                sigma=2.0,
                be=be,
            )
            np.testing.assert_allclose(
                g, np.array([[0.710595, 0.297229], [0.656297, 1.0]]), atol=1e-5
            )
            assert backend.belongs_to_backend(g)

            # soft-DTW
            d = tslearn.metrics.cdist_soft_dtw(
                cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type), gamma=0.01, be=be
            )
            np.testing.assert_allclose(
                d, np.array([[-0.010986, 1.0], [1.0, 0.0]]), atol=1e-5
            )
            assert backend.belongs_to_backend(d)

            d = tslearn.metrics.cdist_soft_dtw(
                cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type),
                cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type),
                gamma=0.01,
                be=be,
            )
            np.testing.assert_allclose(
                d, np.array([[-0.010986, 1.0], [1.0, 0.0]]), atol=1e-5
            )
            assert backend.belongs_to_backend(d)

            n, sz, d = 15, 10, 3
            rng = np.random.RandomState(0)
            time_series = rng.randn(n, sz, d)
            time_series = cast(time_series, array_type)
            dists = tslearn.metrics.cdist_soft_dtw_normalized(time_series, be=be)
            assert backend.all(dists >= 0)
            assert backend.belongs_to_backend(dists)

            v1 = rng.randn(n, 1)
            v2 = rng.randn(n, 1)
            sqeuc = tslearn.metrics.SquaredEuclidean(
                cast(np.array(v1.flat), array_type),
                cast(np.array(v2.flat), array_type), be=be)
            c_dist = cdist(v1, v2, metric="sqeuclidean")
            sqeuc_compute = sqeuc.compute()
            np.testing.assert_allclose(sqeuc_compute, c_dist, atol=1e-5)
            assert backend.belongs_to_backend(sqeuc_compute)


@pytest.mark.skipif(
    (sys.version_info.major, sys.version_info.minor) == (3, 9)
    and "mac" in platform.platform().lower(),
    reason="Test failing for MacOS with python3.9 (Segmentation fault)",
)
def test_gamma_soft_dtw():
    for be in backends:
        for array_type in array_types:
            dataset = cast([[1, 2, 2, 3], [1.0, 2.0, 3.0, 4.0]], array_type)
            gamma = tslearn.metrics.gamma_soft_dtw(
                dataset=dataset, n_samples=200, random_state=0, be=be
            )
            np.testing.assert_allclose(gamma, 8.0)
            backend = instantiate_backend(be, array_type)
            assert backend.belongs_to_backend(gamma)


@pytest.mark.skipif(
    (sys.version_info.major, sys.version_info.minor) == (3, 9)
    and "mac" in platform.platform().lower(),
    reason="Test failing for MacOS with python3.9 (Segmentation fault)",
)
def test_symmetric_cdist():
    for be in backends:
        for array_type in array_types:
            backend = instantiate_backend(be, array_type)
            rng = np.random.RandomState(0)
            dataset = rng.randn(5, 10, 2)
            dataset = cast(dataset, array_type)
            c_dist_dtw = tslearn.metrics.cdist_dtw(dataset, be=be)
            np.testing.assert_allclose(
                c_dist_dtw,
                tslearn.metrics.cdist_dtw(dataset, dataset, be=be),
            )
            assert backend.belongs_to_backend(c_dist_dtw)
            c_dist_gak = tslearn.metrics.cdist_gak(dataset, be=be)
            np.testing.assert_allclose(
                c_dist_gak,
                tslearn.metrics.cdist_gak(dataset, dataset, be=be),
                atol=1e-5,
            )
            assert backend.belongs_to_backend(c_dist_gak)
            c_dist_soft_dtw = tslearn.metrics.cdist_soft_dtw(dataset, be=be)
            np.testing.assert_allclose(
                c_dist_soft_dtw,
                tslearn.metrics.cdist_soft_dtw(dataset, dataset, be=be),
            )
            assert backend.belongs_to_backend(c_dist_soft_dtw)
            c_dist_soft_dtw_normalized = tslearn.metrics.cdist_soft_dtw_normalized(dataset, be=be)
            np.testing.assert_allclose(
                c_dist_soft_dtw_normalized,
                tslearn.metrics.cdist_soft_dtw_normalized(dataset, dataset, be=be),
            )
            assert backend.belongs_to_backend(c_dist_soft_dtw_normalized)


def test_lb_keogh():
    for be in backends:
        for array_type in array_types:
            ts1 = cast([1, 2, 3, 2, 1], array_type)
            env_low, env_up = tslearn.metrics.lb_envelope(ts1, radius=1, be=be)
            np.testing.assert_allclose(
                env_low, np.array([[1.0], [1.0], [2.0], [1.0], [1.0]])
            )
            np.testing.assert_allclose(
                env_up, np.array([[2.0], [3.0], [3.0], [3.0], [2.0]])
            )
            backend = instantiate_backend(be, array_type)
            assert backend.belongs_to_backend(env_low)
            assert backend.belongs_to_backend(env_up)


def test_dtw_path_from_metric():
    for be in backends:
        for array_type in array_types:
            backend = instantiate_backend(be, array_type)
            rng = np.random.RandomState(0)
            s1 = cast(rng.rand(10, 2), array_type)
            s2 = cast(rng.rand(30, 2), array_type)

            # Use dtw_path as a reference
            path_ref, dist_ref = tslearn.metrics.dtw_path(s1, s2, be=be)

            # Test of using a scipy distance function
            path, dist = tslearn.metrics.dtw_path_from_metric(
                s1, s2, metric="sqeuclidean", be=be
            )
            np.testing.assert_equal(path, path_ref)
            assert isinstance(path, list)
            np.testing.assert_allclose(backend.sqrt(dist), dist_ref)
            assert backend.belongs_to_backend(dist)

            # Test of defining a custom function
            def sqeuclidean(x, y):
                return backend.sum((x - y) ** 2)

            path, dist = tslearn.metrics.dtw_path_from_metric(
                s1, s2, metric=sqeuclidean, be=be
            )
            np.testing.assert_equal(path, path_ref)
            assert isinstance(path, list)
            np.testing.assert_allclose(backend.sqrt(dist), dist_ref)
            assert backend.belongs_to_backend(dist)

            # Test of precomputing the distance matrix
            dist_matrix = cdist(s1, s2, metric="sqeuclidean")
            dist_matrix = cast(dist_matrix, array_type)
            path, dist = tslearn.metrics.dtw_path_from_metric(
                dist_matrix, metric="precomputed", be=be
            )
            np.testing.assert_equal(path, path_ref)
            assert isinstance(path, list)
            np.testing.assert_allclose(backend.sqrt(dist), dist_ref)
            assert backend.belongs_to_backend(dist)


def test_softdtw():
    for be in backends:
        for array_type in array_types:
            rng = np.random.RandomState(0)
            s1 = cast(rng.rand(10, 2), array_type)
            s2 = cast(rng.rand(30, 2), array_type)

            # Use dtw_path as a reference
            path_ref, dist_ref = tslearn.metrics.dtw_path(s1, s2, be=be)
            assert isinstance(path_ref, list)
            backend = instantiate_backend(be, array_type)
            assert backend.belongs_to_backend(dist_ref)
            mat_path_ref = np.zeros((10, 30))
            for i, j in path_ref:
                mat_path_ref[i, j] = 1.0

            # Test of using a scipy distance function
            matrix_path, dist = tslearn.metrics.soft_dtw_alignment(s1, s2, gamma=0.0, be=be)
            assert backend.belongs_to_backend(matrix_path)
            assert backend.belongs_to_backend(dist)

            np.testing.assert_equal(dist, dist_ref**2)
            np.testing.assert_allclose(matrix_path, mat_path_ref)

            ts1 = cast([[0.0]], array_type)
            ts2 = cast([[1.0]], array_type)
            sim = tslearn.metrics.soft_dtw(
                ts1=ts1, ts2=ts2, gamma=1.0, be=be, compute_with_backend=True
            )
            assert sim == 1.0


def test_dtw_path_with_empty_or_nan_inputs():
    for be in backends:
        for array_type in array_types:
            s1 = cast(np.zeros((3, 10)), array_type)
            s2_empty = cast(np.zeros((0, 10)), array_type)
            with pytest.raises(ValueError) as excinfo:
                dtw_path(s1, s2_empty, be=be)
            assert (
                str(excinfo.value)
                == "One of the input time series contains only nans or has zero length."
            )

            s2_nan = cast(np.full((3, 10), np.nan), array_type)
            with pytest.raises(ValueError) as excinfo:
                dtw_path(s1, s2_nan, be=be)
            assert (
                str(excinfo.value)
                == "One of the input time series contains only nans or has zero length."
            )


@pytest.mark.skipif(
    len(backends) == 1,
    reason="Skipping test that requires pytorch backend",
)
def test_soft_dtw_loss_pytorch():
    """Tests for the class SoftDTWLossPyTorch."""
    from tslearn.metrics.soft_dtw_loss_pytorch import _SoftDTWLossPyTorch

    b = 5
    m = 10
    n = 12
    d = 8
    batch_ts_1 = torch.zeros((b, m, d), requires_grad=True)
    batch_ts_2 = torch.ones((b, n, d), requires_grad=True)
    soft_dtw_loss_pytorch = tslearn.metrics.SoftDTWLossPyTorch(
        gamma=1.0, normalize=False, dist_func=None
    )
    loss = soft_dtw_loss_pytorch.forward(batch_ts_1, batch_ts_2)
    np.testing.assert_allclose(loss.detach().numpy(), 91.9806137 * torch.ones((b,)))

    loss = soft_dtw_loss_pytorch.forward(batch_ts_1, batch_ts_1)
    np.testing.assert_allclose(loss.detach().numpy(), -14.1957006 * torch.ones((b,)))

    soft_dtw_loss_pytorch_normalized = tslearn.metrics.SoftDTWLossPyTorch(
        gamma=1.0, normalize=True, dist_func=None
    )
    loss_normalized = soft_dtw_loss_pytorch_normalized.forward(batch_ts_1, batch_ts_1)
    assert torch.all(
        loss_normalized
        == torch.zeros(
            b,
        )
    )

    loss_normalized = soft_dtw_loss_pytorch_normalized.forward(batch_ts_1, batch_ts_2)
    np.testing.assert_allclose(loss_normalized.detach().numpy(), 107.89006805 * torch.ones((b,)))

    def euclidean_abs_dist(x, y):
        """Calculates the Euclidean squared distance between each element in x and y per timestep.

        Parameters
        ----------
        x : Tensor, shape=[b, m, d]
            Batch of time series.
        y : Tensor, shape=[b, n, d]
            Batch of time series.

        Returns
        -------
        dist : Tensor, shape=[b, m, n]
            The pairwise squared Euclidean distances.
        """
        m = x.size(1)
        n = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, m, n, d)
        y = y.unsqueeze(1).expand(-1, m, n, d)
        return torch.abs(x - y).sum(3)

    soft_dtw_loss_pytorch = tslearn.metrics.SoftDTWLossPyTorch(
        gamma=1.0, normalize=False, dist_func=euclidean_abs_dist
    )
    loss = soft_dtw_loss_pytorch.forward(batch_ts_1, batch_ts_2)
    np.testing.assert_allclose(loss.detach().numpy(), 91.9806137 * torch.ones((b,)))

    def soft_dtw_loss_function(x, y, dist_func, gamma):
        d_xy = dist_func(x, y)
        return _SoftDTWLossPyTorch.apply(d_xy, gamma)

    loss = soft_dtw_loss_function(
        x=batch_ts_1, y=batch_ts_2, dist_func=euclidean_abs_dist, gamma=0.1
    )
    loss.mean().backward()

    expected_grad_ts1 = np.repeat(
        np.repeat(
            np.reshape(
                np.array(
                    [
                        -0.244445577,
                        -0.244445756,
                        -0.244445294,
                        -0.244443387,
                        -0.244441003,
                        -0.244441897,
                        -0.244444370,
                        -0.244441718,
                        -0.244441971,
                        -0.200000003,
                    ]
                ),
                (1, m, 1),
            ),
            repeats=b,
            axis=0,
        ),
        repeats=d,
        axis=2,
    )
    np.testing.assert_allclose(batch_ts_1.grad, expected_grad_ts1, rtol=5e-5)

    expected_grad_ts2 = np.repeat(
        np.repeat(
            np.reshape(
                np.array(
                    [
                        0.2000008374,
                        0.2000008225,
                        0.2000010014,
                        0.2000009865,
                        0.1999993920,
                        0.1999970675,
                        0.1999994069,
                        0.1999980807,
                        0.1999983191,
                        0.1999950409,
                        0.2000000030,
                        0.2000000030,
                    ]
                ),
                (1, n, 1),
            ),
            repeats=b,
            axis=0,
        ),
        repeats=d,
        axis=2,
    )
    np.testing.assert_allclose(batch_ts_2.grad, expected_grad_ts2, rtol=5e-5)
