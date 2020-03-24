import numpy as np
import pytest
from sklearn.model_selection import cross_validate

from tslearn.utils import to_time_series

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_shapelets():
    pytest.importorskip('keras')
    from tslearn.shapelets import ShapeletModel

    n, sz, d = 15, 10, 2
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    y = rng.randint(2, size=n)
    clf = ShapeletModel(n_shapelets_per_size={2: 5},
                        max_iter=1,
                        verbose=0,
                        optimizer="sgd",
                        random_state=0)
    clf.fit(time_series, y)
    np.testing.assert_allclose(clf.shapelets_[0],
                               np.array([[0.56373, 0.494684],
                                         [1.235707, 1.119235]]),
                               atol=1e-2)
    np.testing.assert_allclose(clf.predict(time_series),
                               np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                         1, 0]))

    cross_validate(clf, time_series, y, cv=2)

    model = ShapeletModel(n_shapelets_per_size={3: 2, 4: 1},
                          max_iter = 1)
    model.fit(time_series, y)
    for shp, shp_bis in zip(model.shapelets_,
                            model.shapelets_as_time_series_):
        np.testing.assert_allclose(shp,
                                   to_time_series(shp_bis, remove_nans=True))

    # Test set_weights / get_weights
    clf = ShapeletModel(n_shapelets_per_size={2: 5},
                        max_iter=1,
                        verbose=0,
                        random_state=0)
    clf.fit(time_series, y)
    preds_before = clf.predict_proba(time_series)
    weights = clf.get_weights()
    # Change number of iterations, then refit, then set weights
    clf.max_iter *= 2
    clf.fit(time_series, y)
    clf.set_weights(weights)
    np.testing.assert_allclose(preds_before,
                               clf.predict_proba(time_series))


def test_serializable_shapelets():
    pytest.importorskip('keras')
    from tslearn.shapelets import SerializableShapeletModel

    n, sz, d = 15, 10, 2
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    y = rng.randint(2, size=n)
    clf = SerializableShapeletModel(n_shapelets_per_size={2: 5},
                                    max_iter=1,
                                    verbose=0,
                                    learning_rate=0.01,
                                    random_state=0)
    clf.fit(time_series, y)
    np.testing.assert_allclose(clf.shapelets_[0],
                               np.array([[0.563342, 0.494981],
                                         [1.236804, 1.11963]]),
                               atol=1e-2)
    np.testing.assert_allclose(clf.predict(time_series),
                               np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                         1, 0]))

    params = clf.get_params(deep=True)
    for s1, s2 in zip(sorted(params.keys()),
                      sorted(['batch_size', 'learning_rate', 'max_iter',
                              'n_shapelets_per_size', 'random_state',
                              'total_lengths', 'shapelet_length', 'verbose',
                              'verbose_level', 'weight_regularizer'])):
        np.testing.assert_string_equal(s1, s2)

    from sklearn.model_selection import cross_validate
    cross_validate(clf, time_series, y, cv=2)
