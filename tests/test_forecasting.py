import numpy as np

import pytest

from sklearn.base import clone

from tslearn.generators import random_walks
from tslearn.forecasting import VARIMA, AutoVARIMA, ScaledForecastingPipeline
from tslearn.preprocessing import (
    TimeSeriesScalerMeanVariance,
    TimeSeriesScalerMinMax,
    TimeSeriesResampler,
)


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
        AutoVARIMA(max_d=0, default_d_for_non_stationarity=None).fit(data)

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


def test_verbosity(capteesys):
    data = random_walks(n_ts=10, sz=100, random_state=0)

    VARIMA(0, 0, 1, max_iter=2, verbose=True).fit(data)
    captured = capteesys.readouterr()
    assert "iteration 1, nlle 2944.90830900361\n" in captured.out

    AutoVARIMA(max_iter=2, verbose=1).fit(data)
    captured = capteesys.readouterr()
    assert "Selected d: 1.\n" in captured.out
    assert "Testing model VARIMA" in captured.out
    assert "iteration" not in captured.out
    assert "Computed AIC" in captured.out

    AutoVARIMA(max_iter=2, verbose=2).fit(data)
    captured = capteesys.readouterr()
    assert "Selected d: 1.\n" in captured.out
    assert "Testing model VARIMA" in captured.out
    assert "iteration 1" in captured.out
    assert "Computed AIC" in captured.out

    AutoVARIMA(max_d=0, max_iter=2, verbose=1).fit(data)
    captured = capteesys.readouterr()
    assert "Default d for non stationarity 0 is used." in captured.out


@pytest.mark.parametrize(
    "scaler",
    [
        TimeSeriesScalerMeanVariance(per_timeseries=False),
        TimeSeriesScalerMinMax(per_timeseries=False),
    ]
)
def test_scaled_forecaster_global_matches_manual_scaling(scaler):
    data = random_walks(n_ts=5, sz=20, d=2, random_state=0) * 100 + 500

    pipeline = ScaledForecastingPipeline(VARIMA(1, 0, 0), scaler=scaler).fit(data)
    assert pipeline.per_timeseries_ is False
    predicted = pipeline.predict(n=3)
    assert predicted.shape == (5, 3, 2)

    reference_scaler = clone(scaler)
    scaled_data = reference_scaler.fit_transform(data)
    reference_predicted = VARIMA(1, 0, 0).fit(scaled_data).predict(n=3)
    np.testing.assert_allclose(
        predicted,
        reference_scaler.inverse_transform(reference_predicted)
    )


@pytest.mark.parametrize(
    "scaler",
    [None, TimeSeriesScalerMeanVariance(), TimeSeriesScalerMinMax()]
)
def test_scaled_forecaster_per_series_matches_manual_scaling(scaler):
    # `per_timeseries=True` is the default of tslearn scalers, so both the
    # default `scaler=None` and an explicit scaler exercise per-series mode.
    data = random_walks(n_ts=5, sz=20, d=2, random_state=0)
    # Give series wildly different levels/spreads, the case per-series
    # scaling is meant for.
    data = data * np.array([1, 100, 0.01, 10, 1000]).reshape(5, 1, 1)

    pipeline = ScaledForecastingPipeline(VARIMA(1, 0, 0), scaler=scaler).fit(data)
    assert pipeline.per_timeseries_ is True
    assert len(pipeline.scalers_) == 5
    predicted = pipeline.predict(n=3)
    assert predicted.shape == (5, 3, 2)

    template = pipeline._scaler_template_
    scalers = [clone(template).fit(data[i:i + 1]) for i in range(5)]
    scaled_data = np.concatenate(
        [s.transform(data[i:i + 1]) for i, s in enumerate(scalers)], axis=0
    )
    reference_predicted_scaled = VARIMA(1, 0, 0).fit(scaled_data).predict(n=3)
    reference_predicted = np.concatenate(
        [
            s.inverse_transform(reference_predicted_scaled[i:i + 1])
            for i, s in enumerate(scalers)
        ],
        axis=0,
    )
    np.testing.assert_allclose(predicted, reference_predicted)


def test_scaled_forecaster_per_series_predict_uses_fresh_statistics():
    data = random_walks(n_ts=3, sz=20, d=1, random_state=0)
    data = data * np.array([1, 100, 0.01]).reshape(3, 1, 1)
    pipeline = ScaledForecastingPipeline(VARIMA(1, 0, 0)).fit(data)

    new_data = random_walks(n_ts=3, sz=15, d=1, random_state=1)
    new_data = new_data * np.array([5, 2, 50]).reshape(3, 1, 1) + 3
    predicted_new = pipeline.predict(new_data, n=2)
    assert predicted_new.shape == (3, 2, 1)

    # Statistics used for the new prediction differ from the fit-time ones,
    # since per-series scaling is recomputed from whatever is transformed.
    template = pipeline._scaler_template_
    fresh_scalers = [
        clone(template).fit(new_data[i:i + 1]) for i in range(3)
    ]
    for fitted, fresh in zip(pipeline.scalers_, fresh_scalers):
        assert not np.allclose(fitted.mean_, fresh.mean_)


def test_scaled_forecaster_fit_predict():
    data = random_walks(n_ts=3, sz=20, d=1, random_state=0) * 10 + 50
    pipeline = ScaledForecastingPipeline(VARIMA(1, 0, 0))
    np.testing.assert_allclose(
        pipeline.fit_predict(data, n=3),
        pipeline.fit(data).predict(n=3)
    )


def test_scaled_forecaster_rejects_scaler_without_inverse_transform():
    data = random_walks(n_ts=3, sz=20, random_state=0)
    with pytest.raises(ValueError):
        ScaledForecastingPipeline(
            VARIMA(1, 0, 0),
            scaler=TimeSeriesResampler(sz=20)
        ).fit(data)
