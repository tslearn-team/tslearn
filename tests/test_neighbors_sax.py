import numpy as np
import pytest

from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.piecewise import SymbolicAggregateApproximation


@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('method', ['predict', 'predict_proba', 'kneighbors'])
def test_sax_queries_use_training_transform(scale, method):
    X = np.array([[-3., -2., -1., -2.], [1., 2., 3., 2.]])[..., None]
    params = dict(n_segments=2, alphabet_size_avg=5, scale=scale)
    model = KNeighborsTimeSeriesClassifier(
        n_neighbors=1, metric='sax', metric_params=params).fit(X, [0, 1])
    sax = SymbolicAggregateApproximation(**params).fit(X)
    before = model._sax._get_model_params().copy()
    query = X[[1, 0, 1]]
    predict = getattr(model, method)
    whole = predict(query)
    pieces = [predict(x[None]) for x in query]
    if method == 'kneighbors':
        for i in range(2):
            np.testing.assert_equal(
                whole[i], np.concatenate([p[i] for p in pieces]))
        np.testing.assert_equal(whole[1].ravel(), [1, 0, 1])
    else:
        np.testing.assert_equal(whole, np.concatenate(pieces))
        if method == 'predict':
            np.testing.assert_equal(whole, [1, 0, 1])
    for key, value in before.items():
        np.testing.assert_equal(getattr(model._sax, key), value)
    np.testing.assert_equal(model._sax.transform(query), sax.transform(query))


@pytest.mark.parametrize('fmt', ['json', 'pickle', 'hdf5'])
@pytest.mark.parametrize('scale', [False, True])
def test_sax_training_state_roundtrip(tmp_path, fmt, scale):
    X = np.array([[-3., -2., -1., -2.], [1., 2., 3., 2.]])[..., None]
    model = KNeighborsTimeSeriesClassifier(
        n_neighbors=1, metric='sax',
        metric_params=dict(n_segments=2, alphabet_size_avg=5, scale=scale)
    ).fit(X, [0, 1])
    path = tmp_path / ('model.' + fmt)
    getattr(model, 'to_' + fmt)(path)
    restored = getattr(type(model), 'from_' + fmt)(path)
    for x, expected in zip(X, [0, 1]):
        np.testing.assert_equal(restored.predict(x[None]), [expected])
        np.testing.assert_equal(restored.predict_proba(x[None]),
                                model.predict_proba(x[None]))
    for key, value in model._sax._get_model_params().items():
        np.testing.assert_equal(getattr(restored._sax, key), value)


def test_sax_refit_updates_transform_parameters():
    X = np.array([[-3., -2., -1., -2.], [1., 2., 3., 2.]])[..., None]
    model = KNeighborsTimeSeriesClassifier(
        n_neighbors=1, metric='sax',
        metric_params=dict(n_segments=2, alphabet_size_avg=5, scale=True)
    ).fit(X, [0, 1])
    params = dict(n_segments=4, alphabet_size_avg=3, scale=True)
    model.set_params(metric_params=params).fit(X + 10, [0, 1])
    sax = SymbolicAggregateApproximation(**params).fit(X + 10)
    np.testing.assert_equal(model._ts_fit, sax.transform(X + 10))
    np.testing.assert_equal(model._sax.mu_, sax.mu_)
    np.testing.assert_equal(model.predict((X + 10)[1:]), [1])


@pytest.mark.parametrize('fmt', ['json', 'pickle', 'hdf5'])
@pytest.mark.parametrize('params', [None, dict(scale=True)])
def test_sax_roundtrip_default_parameters(tmp_path, fmt, params):
    X = np.random.RandomState(0).randn(10, 20, 2)
    model = KNeighborsTimeSeriesClassifier(
        n_neighbors=1, metric='sax', metric_params=params
    ).fit(X, np.arange(10))
    path = tmp_path / ('model.' + fmt)
    getattr(model, 'to_' + fmt)(path)
    restored = getattr(type(model), 'from_' + fmt)(path)
    np.testing.assert_equal(
        restored.predict(X[::2]), model.predict(X[::2]))
    np.testing.assert_equal(
        restored._sax.transform(X), model._sax.transform(X))
