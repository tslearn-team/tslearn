import warnings
from warnings import catch_warnings

import numpy as np

import pytest

from sklearn.base import BaseEstimator, ClassifierMixin

from tslearn.early_classification import NonMyopicEarlyClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.utils import to_time_series_dataset


def test_NonMyopicEarlyClassifier():

    dataset = to_time_series_dataset(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [3, 2, 1, 1, 2, 3],
            [3, 2, 1, 1, 2, 3],
        ]
    )

    y = [0, 0, 0, 1, 1, 1, 0, 0]
    model = NonMyopicEarlyClassifier(
        n_clusters=3,
        base_classifier=KNeighborsTimeSeriesClassifier(
            n_neighbors=1, metric="euclidean"
        ),
        min_t=2,
        lamb=1000.0,
        cost_time_parameter=0.1,
        random_state=0,
    )
    assert model.classes_ is None
    model.fit(dataset, y)
    np.testing.assert_almost_equal(model.early_classification_cost(dataset, y), 0.35)

    # Fewer timestamps than min_ts
    pred, delays = model.early_predict(dataset[:, :1])
    np.testing.assert_array_equal(pred, np.array([np.nan] * 8))
    np.testing.assert_array_equal(delays, np.array([np.nan] * 8))

    pred, delays = model.early_predict(dataset[:, :3])
    np.testing.assert_array_equal(pred, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(delays, np.array([1, 1, 1, 1, 1, 1, 0, 0]))

    pred, delays = model.early_predict_proba(dataset[:, :3])
    np.testing.assert_array_equal(
        pred,
        np.array([[1.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 0.0],
                  [1.0, 0.0]])
    )
    np.testing.assert_array_equal(delays, np.array([1, 1, 1, 1, 1, 1, 0, 0]))

    # More timestamps than trained dataset
    data = to_time_series_dataset([[1, 2, 3, 3, 2, 1, 1, 2, 3]])
    with pytest.raises(ValueError):
        model.early_predict(data)

    data = to_time_series_dataset([[1, 2, 3, 3, 2, 1]])
    gen = model.get_early_predict_generator()
    expected_preds = np.array([[np.nan], [0], [0], [1], [1], [1]])
    expected_delays = np.array([[np.nan], [2], [1], [0], [0], [0], [0]])
    for i in range(data.shape[1]):
        pred, delay = gen.send(data[:, i:i+1, :])
        np.testing.assert_array_equal(pred, expected_preds[i])
        np.testing.assert_array_equal(delay, expected_delays[i])

    data = to_time_series_dataset([[1, 2, 3, 3, 2, 1]])
    gen = model.get_early_predict_proba_generator()
    expected_preds = np.array([
        [[np.nan, np.nan]],
        [[1.0, 0.0]],
        [[1.0, 0.0]],
        [[0.0, 1.0]],
        [[0.0, 1.0]],
        [[0.0, 1.0]]
    ])
    expected_delays = np.array([[np.nan], [2], [1], [0], [0], [0], [0]])
    for i in range(data.shape[1]):
        pred, delay = gen.send(data[:, i:i+1, :])
        np.testing.assert_array_equal(pred, expected_preds[i])
        np.testing.assert_array_equal(delay, expected_delays[i])

    # Check unproperly formatted generator input
    with pytest.warns(RuntimeWarning):
        gen.send(1)

    # Check iteration raises after n_samples + 1
    with pytest.raises(ValueError):
        gen.send([[[1]]])


class _AlwaysOneClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.ones(len(X), dtype=int)


def test_NonMyopicEarlyClassifier_misclassification_probabilities():
    # pyhatyck_[t, k, y, y_hat] is P(y_hat | y, c_k) for y != y_hat, so it
    # must be normalized over the predicted classes for each true class.
    dataset = to_time_series_dataset(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [1, 2, 3, 3, 2, 1],
            [3, 2, 1, 1, 2, 3],
            [3, 2, 1, 1, 2, 3],
        ]
    )
    y = [0, 0, 0, 1, 1, 1, 0, 0]
    # This classifier predicts class 1 for every series: class 0 series are
    # always misclassified and class 1 series never are.
    model = NonMyopicEarlyClassifier(
        n_clusters=1,
        base_classifier=_AlwaysOneClassifier(),
        random_state=0,
    ).fit(dataset, y)
    np.testing.assert_allclose(model.pyhatyck_[:, 0, 0, 1], 1.0)
    np.testing.assert_allclose(model.pyhatyck_[:, 0, 1], 0.0)
