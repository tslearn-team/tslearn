import numpy as np

import tslearn.barycenters

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_set_weights():
    w = tslearn.barycenters._set_weights(None, 3)
    np.testing.assert_allclose(w, np.ones((3, )))

    w = tslearn.barycenters._set_weights([.5, .25, .25], 3)
    np.testing.assert_allclose(w, [.5, .25, .25])

    w = tslearn.barycenters._set_weights([.5, .25, .25], 2)
    np.testing.assert_allclose(w, np.ones((2, )))


def test_euclidean_barycenter():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    bar = tslearn.barycenters.euclidean_barycenter(time_series)
    np.testing.assert_allclose(bar, time_series.mean(axis=0))

    weights = rng.rand(n, )
    weights /= np.sum(weights)
    bar = tslearn.barycenters.euclidean_barycenter(time_series,
                                                   weights=weights)
    np.testing.assert_allclose(bar, np.average(time_series,
                                               axis=0, weights=weights))


def test_dba():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)

    # Equal length, 0 iterations -> Euclidean
    euc_bar = tslearn.barycenters.euclidean_barycenter(time_series)
    dba_bar = tslearn.barycenters.dtw_barycenter_averaging_petitjean(
        time_series,
        max_iter=0)
    np.testing.assert_allclose(euc_bar, dba_bar)

    # Equal length, >0 iterations
    dba_bar = tslearn.barycenters.dtw_barycenter_averaging_petitjean(
        time_series,
        max_iter=5)
    ref = np.array([[0.33447722, 0.0418787, -0.03953774],
                    [-0.75757987, -0.26841384, -0.22418874],
                    [-0.0473153, 0.41030073, 0.06069343],
                    [0.36250957, -0.79033572, 0.02300398],
                    [0.02783764, -0.05039364, -0.79595523],
                    [0.38139685, 0.37661911, 0.09506468],
                    [0.48628337, 0.17192078, -1.16404917],
                    [-0.40263459, 0.59364783, -0.6843561],
                    [0.67493146, -0.37714421, 0.16604165],
                    [-0.32249566, 0.09109832, 0.55489214]])
    np.testing.assert_allclose(dba_bar, ref, atol=1e-6)

    dba_bar = tslearn.barycenters.dtw_barycenter_averaging_petitjean(
        time_series,
        max_iter=5)
    dba_bar_mm = tslearn.barycenters.dtw_barycenter_averaging(
        time_series,
        max_iter=5)
    np.testing.assert_allclose(dba_bar, dba_bar_mm)

    weights = rng.rand(n)
    dba_bar = tslearn.barycenters.dtw_barycenter_averaging_petitjean(
        time_series,
        weights=weights,
        max_iter=5)
    dba_bar_mm = tslearn.barycenters.dtw_barycenter_averaging(
        time_series,
        weights=weights,
        max_iter=5)
    np.testing.assert_allclose(dba_bar, dba_bar_mm)


def test_softdtw_barycenter():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)

    # Equal length, 0 iterations -> Euclidean
    euc_bar = tslearn.barycenters.euclidean_barycenter(time_series)
    sdtw_bar = tslearn.barycenters.softdtw_barycenter(time_series, max_iter=0)
    np.testing.assert_allclose(euc_bar, sdtw_bar)

    # Equal length, >0 iterations
    sdtw_bar = tslearn.barycenters.softdtw_barycenter(time_series, max_iter=5)
    ref = np.array([[0.28049395, -0.01190817, -0.06228361],
                    [-0.67097059, -0.10737132, -0.33867808],
                    [0.29380813, 0.0474172, 0.32718516],
                    [0.14438242, -0.56877605, -0.14563386],
                    [0.15620728, -0.04473494, -0.63912905],
                    [0.35989018, 0.42118863, -0.2127066],
                    [0.16831249, 0.65420655, -0.53587191],
                    [-0.20737107, 0.15301328, -0.74052802],
                    [0.53149515, -0.24839857, -0.03430969],
                    [-0.17690603, 0.07217633, 0.58071408]])
    np.testing.assert_allclose(sdtw_bar, ref, atol=1e-6)
