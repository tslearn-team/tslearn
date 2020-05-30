import numpy as np

from tslearn.piecewise import OneD_SymbolicAggregateApproximation, \
    SymbolicAggregateApproximation, PiecewiseAggregateApproximation
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.exceptions import NotFittedError
from sklearn.base import clone

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_paa():
    unfitted_paa = PiecewiseAggregateApproximation(n_segments=3)
    data = [[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]]
    np.testing.assert_raises(NotFittedError, unfitted_paa.distance,
                             data[0], data[1])

    paa_est = unfitted_paa
    n, sz, d = 2, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    paa_repr = paa_est.fit_transform(X)
    np.testing.assert_allclose(paa_est.distance(X[0], X[1]),
                               paa_est.distance_paa(paa_repr[0], paa_repr[1]))


def test_sax():
    unfitted_sax = SymbolicAggregateApproximation(n_segments=3,
                                                  alphabet_size_avg=2)
    data = [[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]]
    np.testing.assert_raises(NotFittedError, unfitted_sax.distance,
                             data[0], data[1])

    sax_est_no_scale = unfitted_sax
    sax_est_scale = clone(sax_est_no_scale)
    sax_est_scale.set_params(scale=True)
    n, sz, d = 2, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    for sax_est in [sax_est_no_scale, sax_est_scale]:
        sax_repr = sax_est.fit_transform(X)
        np.testing.assert_allclose(
            sax_est.distance(X[0], X[1]),
            sax_est.distance_sax(sax_repr[0], sax_repr[1])
        )


def test_1dsax():
    unfitted_1dsax = OneD_SymbolicAggregateApproximation(n_segments=3,
                                                         alphabet_size_avg=2,
                                                         alphabet_size_slope=2)
    data = [[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]]
    np.testing.assert_raises(NotFittedError, unfitted_1dsax.distance,
                             data[0], data[1])

    sax1d_est_no_scale = unfitted_1dsax
    sax1d_est_scale = clone(sax1d_est_no_scale)
    sax1d_est_scale.set_params(scale=True)
    n, sz, d = 2, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    for sax1d_est in [sax1d_est_no_scale, sax1d_est_scale]:
        sax1d = sax1d_est.fit_transform(X)
        np.testing.assert_allclose(
            sax1d_est.distance(X[0], X[1]),
            sax1d_est.distance_1d_sax(sax1d[0], sax1d[1])
        )


def test_sax_scale():
    n, sz, d = 10, 10, 3
    rng = np.random.RandomState(0)
    X = rng.rand(n, sz, d)
    y = rng.choice([0, 1], size=n)

    sax = SymbolicAggregateApproximation(n_segments=3,
                                         alphabet_size_avg=2,
                                         scale=True)
    sax.fit(X)
    np.testing.assert_array_almost_equal(X,
                                         sax._unscale(sax._scale(X)))

    np.testing.assert_array_almost_equal(np.zeros((d, )),
                                         sax._scale(X).reshape((-1, d)).mean())
    np.testing.assert_array_almost_equal(np.ones((d, )),
                                         sax._scale(X).reshape((-1, d)).std())

    # Case of kNN-SAX
    knn_sax = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="sax",
                                             metric_params={"scale": True})
    knn_sax.fit(X, y)
    X_scale_unscale = knn_sax._sax._unscale(knn_sax._sax._scale(X))
    np.testing.assert_array_almost_equal(X, X_scale_unscale)

    knn_sax.predict(X)
