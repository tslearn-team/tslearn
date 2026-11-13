import numpy as np

import pytest

from scipy.stats import norm

from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tslearn.piecewise import (
    OneD_SymbolicAggregateApproximation,
    SymbolicAggregateApproximation,
    PiecewiseAggregateApproximation
)
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

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


def test_paa_length_not_multiple_of_n_segments():
    # 19 time steps in 10 segments: segments of 1 or 2 steps that cover the
    # whole series
    x = np.arange(19, dtype=float).reshape(1, 19, 1)
    paa = PiecewiseAggregateApproximation(n_segments=10).fit(x)
    x_paa = paa.transform(x)
    np.testing.assert_allclose(x_paa.ravel(),
                               [0.] + [2 * i + 1.5 for i in range(9)])
    np.testing.assert_allclose(
        paa.inverse_transform(x_paa).ravel(),
        [0.] + [2 * i + 1.5 for i in range(9) for _ in range(2)]
    )


def test_sax_inverse_transform_length_not_multiple_of_n_segments():
    # 19 time steps in 10 segments: the first segment has 1 time step, the
    # others 2, and each time step takes the value of its segment's symbol
    rng = np.random.RandomState(0)
    X = rng.randn(2, 19, 1)
    sax = SymbolicAggregateApproximation(n_segments=10,
                                         alphabet_size_avg=5).fit(X)
    X_sax = sax.transform(X)
    expected = np.repeat(sax.breakpoints_avg_middle_[X_sax[..., 0]],
                         [1] + [2] * 9, axis=1)
    np.testing.assert_allclose(sax.inverse_transform(X_sax)[..., 0],
                               expected)


@pytest.mark.parametrize("sz, n_segments", [(19, 10), (60, 16), (150, 64)])
def test_piecewise_distances_lower_bound_euclidean(sz, n_segments):
    rng = np.random.RandomState(0)
    X = TimeSeriesScalerMeanVariance().fit_transform(
        rng.randn(40, sz, 1).cumsum(axis=1)
    )
    paa = PiecewiseAggregateApproximation(n_segments=n_segments).fit(X)
    sax = SymbolicAggregateApproximation(n_segments=n_segments,
                                         alphabet_size_avg=10).fit(X)
    X_paa = paa.transform(X)
    X_sax = sax.transform(X)
    for i in range(0, len(X), 2):
        euclidean = np.linalg.norm(X[i] - X[i + 1])
        assert paa.distance_paa(X_paa[i], X_paa[i + 1]) <= euclidean + 1e-9
        assert sax.distance_sax(X_sax[i], X_sax[i + 1]) <= euclidean + 1e-9


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


def test_piecewise_distance_variable_length():
    # distance() should work with two series of different lengths, the same
    # way transform() already does on a NaN-padded dataset
    rng = np.random.RandomState(0)
    ts1, ts2 = rng.randn(12), rng.randn(9)
    X = to_time_series_dataset([ts1, ts2])

    paa = PiecewiseAggregateApproximation(n_segments=3)
    paa_repr = paa.fit_transform(X)
    np.testing.assert_allclose(paa.distance(ts1, ts2),
                               paa.distance_paa(paa_repr[0], paa_repr[1]))

    sax = SymbolicAggregateApproximation(n_segments=3, alphabet_size_avg=4)
    sax_repr = sax.fit_transform(X)
    np.testing.assert_allclose(sax.distance(ts1, ts2),
                               sax.distance_sax(sax_repr[0], sax_repr[1]))

    sax1d = OneD_SymbolicAggregateApproximation(n_segments=3,
                                                alphabet_size_avg=4,
                                                alphabet_size_slope=3,
                                                sigma_l=1.)
    sax1d_repr = sax1d.fit_transform(X)
    np.testing.assert_allclose(
        sax1d.distance(ts1, ts2),
        sax1d.distance_1d_sax(sax1d_repr[0], sax1d_repr[1])
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


@pytest.mark.parametrize(
    "model_class",
    [PiecewiseAggregateApproximation,
     SymbolicAggregateApproximation,
     OneD_SymbolicAggregateApproximation]
)
def test_fitted_inverse_transform(model_class):
    n, sz, d = 2, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)

    model = model_class()
    with pytest.raises(NotFittedError):
        model.inverse_transform(X)

    transformed = model.fit_transform(X)
    assert X.shape == model.inverse_transform(transformed).shape


def test_1dsax_scale_matches_sax_average():
    # With scale=True the average symbols of 1d-SAX must match those of SAX:
    # both scale the series once, using the same fitted mean and std.
    n, sz, d = 4, 12, 1
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d) * 5 + 10

    sax = SymbolicAggregateApproximation(n_segments=4, alphabet_size_avg=5,
                                         scale=True)
    one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=4,
                                                    alphabet_size_avg=5,
                                                    alphabet_size_slope=5,
                                                    scale=True)
    np.testing.assert_array_equal(sax.fit_transform(X),
                                  one_d_sax.fit_transform(X)[:, :, :d])


@pytest.mark.parametrize("segment_size", [1, 2, 3, 4])
@pytest.mark.parametrize("n_segments", [1, 3])
@pytest.mark.parametrize("n_features", [1, 2])
@pytest.mark.parametrize("average_pair", [(1, 1), (0, 2)])
@pytest.mark.parametrize("slope_pair", [(0, 2), (1, 1)])
def test_1dsax_distance_segment_center(segment_size, n_segments, n_features,
                                       average_pair, slope_pair):
    original_size = segment_size * n_segments
    sax = OneD_SymbolicAggregateApproximation(
        n_segments=n_segments, alphabet_size_avg=3,
        alphabet_size_slope=3, sigma_l=1.5
    ).fit(np.zeros((2, original_size, n_features)))
    symbols = np.empty((2, n_segments, 2 * n_features), dtype=int)
    for i in range(2):
        symbols[i, :, :n_features] = average_pair[i]
        symbols[i, :, n_features:] = slope_pair[i]

    # The discrete segment has mean time (L - 1) / 2, so the cross term
    # vanishes and the squared distance separates into mean and slope terms.
    medians = norm.ppf([1. / 6, 1. / 2, 5. / 6])
    delta_avg = medians[average_pair[0]] - medians[average_pair[1]]
    delta_slope = 1.5 * (medians[slope_pair[0]] - medians[slope_pair[1]])
    expected_sq = n_segments * n_features * (
        segment_size * delta_avg ** 2
        + segment_size * (segment_size ** 2 - 1) / 12 * delta_slope ** 2
    )
    expected = np.sqrt(expected_sq)
    np.testing.assert_allclose(
        sax.distance_1d_sax(symbols[0], symbols[1]), expected
    )
    reconstructed = sax.inverse_transform(symbols)
    np.testing.assert_allclose(
        np.linalg.norm(reconstructed[0] - reconstructed[1]), expected
    )


def test_1dsax_distance_opposite_trends():
    X = np.array([[[0.], [1.], [2.], [3.], [4.], [5.]],
                  [[6.], [5.], [4.], [3.], [2.], [1.]]])
    sax = OneD_SymbolicAggregateApproximation(
        n_segments=2, alphabet_size_avg=3,
        alphabet_size_slope=3, sigma_l=1.
    )
    symbols = sax.fit_transform(X)
    reconstructed = sax.inverse_transform(symbols)
    expected = np.linalg.norm(reconstructed[0] - reconstructed[1])
    np.testing.assert_allclose(sax.distance(X[0], X[1]), expected)
