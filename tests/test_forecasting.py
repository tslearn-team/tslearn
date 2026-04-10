import numpy as np

import pytest

from tslearn.generators import random_walks
from tslearn.forecasting import VARIMA, AutoVARIMA


def test_VARIMA():
    # random walk with std = 0 is constant
    data = random_walks(n_ts=100, sz=10, std=0)

    # Test fit accepts variable length dataset,
    data[0, 6:, :] = np.nan
    model = VARIMA(1, 0, 0).fit(data)
    predicted = model.predict()
    np.testing.assert_array_equal(
        predicted,
        data[:, 0, :].reshape(predicted.shape)
    )

    # Test predict accepts variable length dataset
    predict_data = random_walks(n_ts=2, sz=9, std=0)
    predict_data[0, 7:, :] = np.nan
    predicted = model.predict(predict_data)
    np.testing.assert_array_equal(
        predicted,
        predict_data[:, 0, :].reshape(predicted.shape)
    )

    # Test TS min_size at predict / fit
    data[0, 1:, :] = np.nan
    # Should error min_size
    with pytest.raises(ValueError):
        model.predict(data)
    # Should error min_size
    with pytest.raises(ValueError):
        model.fit(data)

    # Test multivariate, variable length
    horizon = 2
    data = random_walks(d=3)
    data[25, 250:, :] = np.nan
    data[75, 120:, :] = np.nan
    predicted = VARIMA(2, 1, 2, with_constant=False).fit_predict(data, n=horizon)
    assert predicted.shape == (data.shape[0], horizon, data.shape[-1])

    # Test p, q, d = 0 without constant, should predict 0's
    horizon = 5
    data = random_walks(d=3)
    model = VARIMA(0, 0, 0, with_constant=False).fit(data)
    predicted = model.predict(n=horizon)
    np.testing.assert_array_equal(
        predicted,
        np.zeros((data.shape[0], horizon, data.shape[-1]))
    )

    # Test p, q, d = 0 with constant, should predict constant mean
    horizon = 5
    data = random_walks(d=3)
    model = VARIMA(0, 0, 0, with_constant=True).fit(data)
    predicted = model.predict(n=horizon)
    np.testing.assert_almost_equal(
        predicted,
        np.full((data.shape[0], horizon, data.shape[-1]), np.mean(data, axis=(0, 1)))
    )

    # Univariate x(t+1) = 2x(t)
    data = np.array([
        [[1], [2], [4]],
        [[3], [6], [12]],
        [[2], [4], [8]],
    ])
    model = VARIMA(1, 0, 0).fit(data)
    predicted = model.predict()
    expected = np.array([
        [[8.]],
        [[24]],
        [[16]],
    ])
    np.testing.assert_almost_equal(predicted, expected)

    # Univariate with constant x(t+1) = 0.5x(t) + 0.5
    data = np.array([
        [[4], [2.5], [1.75]],
        [[8], [4.5], [2.75]],
        [[6], [3.5], [2.25]],
    ])
    model = VARIMA(1, 0, 0).fit(data)
    predicted = model.predict()
    expected = np.array([
        [[1.375]],
        [[1.875]],
        [[1.625]],
    ])
    np.testing.assert_almost_equal(predicted, expected)

    # Univariate with constant x(t+1) - x(t) = 0.5(x(t) - x(t-1)) + 0.5 -> x(t+1) = 1.5x(t) - 0.5x(t-1)) + 0.5
    data = np.array([
        [[4], [6.5], [8.25], [9.625]],
        [[8], [12.5], [15.25], [17.125]],
        [[6], [9.5], [11.75], [13.375]],
    ])
    model = VARIMA(1, 1, 0).fit(data)
    predicted = model.predict(n=2)
    expected = np.array([
        [[10.8125], [11.90625]],
        [[18.5625], [19.78125]],
        [[14.6875], [15.84375]],
    ])
    np.testing.assert_almost_equal(predicted, expected)
    np.testing.assert_almost_equal(
        model.predict(data, n=2),
        expected
    )

    # MA X(t) = e(t) + 0.9e(t-1)
    rng = np.random.RandomState(0)
    noise = rng.normal(size=(2, 100, 2))
    data = noise[:, 1:] + 0.9 * noise[:, :-1]
    model = VARIMA(0, 0, 1).fit(data)
    np.testing.assert_allclose(
        model.ma_coeffs_,
        np.array([[[0.9, 0], [0, 0.9]]]),
        atol=0.1
    )
    np.testing.assert_allclose(
        model.predict(n=2),
        model.predict(data, n=2)
    )


def test_AutoVARIMA():
    rng = np.random.RandomState(0)
    data = random_walks(n_ts=10, sz=100, std=0.1, random_state=rng)

    with pytest.raises(ValueError):
        AutoVARIMA(max_d=0,default_d_for_non_stationarity=None).fit(data)

    # Test max orders
    model = AutoVARIMA(max_p=0, max_q=0, max_d=0).fit(data)
    assert model.best_estimator_.p == model.best_estimator_.q == model.best_estimator_.d == 0

    # Non-stationary AR 1
    model = AutoVARIMA().fit(data)
    assert model.best_estimator_.p == model.best_estimator_.q == 0
    assert model.best_estimator_.d == 1

    # Non-stationary AR 1 with max_d = 0
    model = AutoVARIMA(max_d=0).fit(data)
    assert model.best_estimator_.p == 1
    assert model.best_estimator_.d == 0

    # Should error min_size
    model = AutoVARIMA(max_p=0, max_q=0, max_d=0, seasonal_period=5).fit(data)
    with pytest.raises(ValueError):
        model.predict(data[0, :5:])
    # Should error min_size
    with pytest.raises(ValueError):
        model.fit(data[0, :5:])

    # Estimating normally distributed noise
    data = rng.normal(size=(10, 100, 2))
    model = AutoVARIMA().fit(data)
    assert model.best_estimator_.p == model.best_estimator_.q == model.best_estimator_.d == 0

    # Test seasonality with MA X(t) = e(t) + 0.9e(t-1)
    seasonal_period = 10
    noise = rng.normal(size=(2, 100, 2))
    data = noise[:, 1:] + 0.9 * noise[:, :-1]
    seasonal_data = np.cos(np.linspace(0, 2 * np.pi * data.shape[1] / seasonal_period, data.shape[1]))
    for k in range(data.shape[-1]):
        data[..., k] += seasonal_data
    model = AutoVARIMA(seasonal_period=10).fit(data)
    assert model.best_estimator_.p == model.best_estimator_.d == 0
    assert model.best_estimator_.q == 1
    np.testing.assert_allclose(
        model.predict(n=2),
        model.predict(data, n=2)
    )
    np.testing.assert_allclose(
        model.predict(n=2),
        model.fit_predict(data, n=2)
    )
