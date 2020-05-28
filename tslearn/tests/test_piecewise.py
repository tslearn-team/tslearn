import numpy as np

from tslearn.piecewise import OneD_SymbolicAggregateApproximation, \
    SymbolicAggregateApproximation, PiecewiseAggregateApproximation
from sklearn.exceptions import NotFittedError

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

    sax_est = unfitted_sax
    n, sz, d = 2, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    sax_repr = sax_est.fit_transform(X)
    np.testing.assert_allclose(sax_est.distance(X[0], X[1]),
                               sax_est.distance_sax(sax_repr[0], sax_repr[1]))


def test_1dsax():
    unfitted_1dsax = OneD_SymbolicAggregateApproximation(n_segments=3,
                                                         alphabet_size_avg=2,
                                                         alphabet_size_slope=2)
    data = [[-1., 2., 0.1, -1., 1., -1.], [1., 3.2, -1., -3., 1., -1.]]
    np.testing.assert_raises(NotFittedError, unfitted_1dsax.distance,
                             data[0], data[1])

    sax1d_est = unfitted_1dsax
    n, sz, d = 2, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    sax1d = sax1d_est.fit_transform(X)
    np.testing.assert_allclose(
        sax1d_est.distance(X[0], X[1]),
        sax1d_est.distance_1d_sax(sax1d[0], sax1d[1])
    )
