import numpy as np
from tslearn.neighbors import KNeighborsTimeSeriesClassifier

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_constrained_paths():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    X = rng.randn(n, sz, d)
    y = rng.randint(low=0, high=3, size=n)

    model_euc = KNeighborsTimeSeriesClassifier(n_neighbors=3,
                                               metric="euclidean")
    y_pred_euc = model_euc.fit(X, y).predict(X)
    model_dtw_sakoe = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="dtw",
        metric_params={
            "global_constraint": "sakoe_chiba",
            "sakoe_chiba_radius":0
        }
    )
    y_pred_sakoe = model_dtw_sakoe.fit(X, y).predict(X)
    np.testing.assert_equal(y_pred_euc, y_pred_sakoe)

    model_softdtw = KNeighborsTimeSeriesClassifier(
        n_neighbors=3,
        metric="softdtw",
        metric_params={
            "gamma": 1e-6
        }
    )
    y_pred_softdtw = model_softdtw.fit(X, y).predict(X)

    model_dtw = KNeighborsTimeSeriesClassifier(
            n_neighbors=3,
            metric="dtw"
    )
    y_pred_dtw = model_dtw.fit(X, y).predict(X)

    np.testing.assert_equal(y_pred_dtw, y_pred_softdtw)
