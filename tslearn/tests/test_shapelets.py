import numpy as np
import pytest
from sklearn.model_selection import cross_validate

from tslearn.utils import to_time_series, to_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_shapelets():
    shapelets = pytest.importorskip('tslearn.shapelets')
    from keras.optimizers import Adam

    n, sz, d = 15, 10, 2
    rng = np.random.RandomState(0)
    time_series = rng.randn(n, sz, d)
    y = rng.randint(2, size=n)
    clf = shapelets.LearningShapelets(n_shapelets_per_size={2: 5},
                            max_iter=1,
                            verbose=0,
                            optimizer="sgd",
                            random_state=0)

    cross_validate(clf, time_series, y, cv=2)

    clf = shapelets.LearningShapelets(n_shapelets_per_size={2: 5},
                            max_iter=1,
                            verbose=0,
                            optimizer=Adam(learning_rate=.1),
                            random_state=0)
    cross_validate(clf, time_series, y, cv=2)

    model = shapelets.LearningShapelets(n_shapelets_per_size={3: 2, 4: 1}, max_iter=1)
    model.fit(time_series, y)
    for shp, shp_bis in zip(model.shapelets_,
                            model.shapelets_as_time_series_):
        np.testing.assert_allclose(shp,
                                   to_time_series(shp_bis, remove_nans=True))

    # Test set_weights / get_weights
    clf = shapelets.LearningShapelets(n_shapelets_per_size={2: 5},
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

    clf = shapelets.LearningShapelets(max_iter=1,
                            verbose=0,
                            random_state=0)
    clf.fit(time_series, y)
    assert clf.shapelets_.shape == (6,)
    assert clf.shapelets_as_time_series_.shape == (6, 3, 2)


def test_shapelet_lengths():
    shapelets = pytest.importorskip('tslearn.shapelets')

    # Test variable-length
    y = [0, 1]
    time_series = to_time_series_dataset([[1, 2, 3, 4, 5], [3, 2, 1]])
    clf = shapelets.LearningShapelets(n_shapelets_per_size={3: 1},
                            max_iter=1,
                            verbose=0,
                            random_state=0)
    clf.fit(time_series, y)

    weights_shapelet = [np.array([[[1], [2], [3]]])]
    clf.set_weights(weights_shapelet, layer_name="shapelets_0")
    tr = clf.transform(time_series)
    np.testing.assert_allclose(tr,
                               np.array([[0.], [8. / 3]]))

    # Test max_size to predict longer series than those passed at fit time
    y = [0, 1]
    time_series = to_time_series_dataset([[1, 2, 3, 4, 5], [3, 2, 1]])
    clf = shapelets.LearningShapelets(n_shapelets_per_size={3: 1},
                            max_iter=1,
                            verbose=0,
                            max_size=6,
                            random_state=0)
    clf.fit(time_series[:, :-1], y)  # Fit with size 4
    weights_shapelet = [np.array([[[1], [2], [3]]])]
    clf.set_weights(weights_shapelet, layer_name="shapelets_0")
    tr = clf.transform(time_series)
    np.testing.assert_allclose(tr,
                               np.array([[0.], [8. / 3]]))


def test_series_lengths():
    shapelets = pytest.importorskip('tslearn.shapelets')

    # Test long shapelets
    y = [0, 1]
    time_series = to_time_series_dataset([[1, 2, 3, 4, 5], [3, 2, 1]])
    clf = shapelets.LearningShapelets(n_shapelets_per_size={8: 1},
                            max_iter=1,
                            verbose=0,
                            random_state=0)
    np.testing.assert_raises(ValueError, clf.fit, time_series, y)

    # Test small max_size
    y = [0, 1]
    time_series = to_time_series_dataset([[1, 2, 3, 4, 5], [3, 2, 1]])
    clf = shapelets.LearningShapelets(n_shapelets_per_size={3: 1},
                            max_iter=1,
                            verbose=0,
                            max_size=4,
                            random_state=0)
    np.testing.assert_raises(ValueError, clf.fit, time_series, y)


def test_locate():
    shapelets = pytest.importorskip('tslearn.shapelets')

    y = [0, 1]
    time_series = to_time_series_dataset([[1, 2, 3, 4, 5], [3, 2, 1]])
    clf = shapelets.LearningShapelets(n_shapelets_per_size={2: 1},
                            max_iter=100,
                            verbose=0,
                            random_state=0)
    clf.fit(time_series, y)
    shapelet = clf.shapelets_[0]
    np.testing.assert_allclose(
        shapelet,
        np.array([[2.6348903], [2.3668802]])
    )

    predicted_locations = clf.locate(time_series)
    distances = []
    for time_serie in time_series:
        distances.append([
            np.linalg.norm(time_serie[i:i + len(shapelet)] - shapelet)
            for i in range(len(time_serie) - len(shapelet))
        ])
    np.testing.assert_array_equal(predicted_locations, np.nanargmin(distances, axis=1, keepdims=True))
