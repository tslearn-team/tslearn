import numpy as np

import pytest

try:
    import torch
    backends = ["numpy", "pytorch", None]
except ImportError:
    backends = ["numpy", None]

from tslearn.utils import to_time_series_dataset
from tslearn.metrics import performance


def test_inputs():

    for be in backends:
        if be is not None:
            y_true = to_time_series_dataset([1, 2, 3], be=be)
            y_pred = to_time_series_dataset([0, 1, 2], be=be)
        else:
            y_true = [1, 2, 3]
            y_pred = [0, 1, 2]

        assert performance.mae(y_true, y_pred) == 1
        assert performance.mae(y_true, y_pred, timestamps_weights=[1, 0, 0]) == 1
        assert performance.mae(y_true, y_pred, multioutput=[1]) == 1
        assert performance.mse(y_true, y_pred) == 1
        assert performance.mse(y_true, y_pred, timestamps_weights=[1, 0, 0]) == 1
        assert performance.mse(y_true, y_pred, multioutput=[1]) == 1
        assert performance.mase(y_true, y_pred, y_true) == 1


def test_mae():
    y_true = [
        [[1, 2], [2, 3], [3, 4]],
        [[1, 2], [2, 3], [3, 4]]
    ]
    y_pred = [
        [[1, 2], [2, 3], [3, 4]],
        [[0, 1], [1, 2], [2, 3]]
    ]
    mae_ = performance.mae(y_pred, y_true)
    assert mae_ == 1/2
    mae_ = performance.mae(y_pred, y_true, ts_weights=[0, 1])
    assert mae_ == 1
    mae_ = performance.mae(y_pred, y_true, timestamps_weights=[0, 1, 0])
    assert mae_ == 1/2
    mae_ = performance.mae(y_pred, y_true, multioutput=[0, 1])
    assert mae_ == 1/2
    mae_ = performance.mae(y_pred, y_true, multioutput="raw_values")
    np.testing.assert_almost_equal(mae_, [1/2, 1/2])
    mae_ = performance.mae(
        y_pred,
        y_true,
        ts_weights=[0, 1],
        timestamps_weights=[0, 1, 0]
    )
    assert mae_ == 1
    mae_ = performance.mae(
        y_pred,
        y_true,
        ts_weights=[0, 1],
        timestamps_weights=[0, 1, 0],
        multioutput=[0, 1]
    )
    assert mae_ == 1
    mae_ = performance.mae(
        y_pred,
        y_true,
        ts_weights=[0, 1],
        timestamps_weights=[0, 1, 0],
        multioutput="raw_values"
    )
    np.testing.assert_almost_equal(mae_, [1, 1])

    rng = np.random.default_rng(0)
    sz = 10000
    y_true = [0] * sz

    # Uniform distribution
    y_pred = np.array(y_true) + rng.uniform(-0.5, 0.5, sz)
    mae_ = performance.mae(y_pred, y_true)
    np.testing.assert_almost_equal(mae_, 0.25, decimal=2)

    # Normal distribution
    y_pred = np.array(y_true) + rng.normal(0, 0.5, sz)
    mae_ = performance.mae(y_pred, y_true)
    np.testing.assert_almost_equal(mae_, np.sqrt(2/np.pi)*0.5, decimal=2)

    assert performance.mae(y_pred, y_true) == performance.mae(y_true, y_pred)


def test_mse():
    y_true = [
        [[0.5, 2], [2, 3], [1, 4]],
        [[1, 2], [2, 3], [3, 4]]
    ]
    y_pred = [
        [[1, 2], [2, 3], [3, 4]],
        [[0, 1], [1, 2], [2, 3]]
    ]

    assert performance.mse(y_pred, y_true) == performance.mse(y_true, y_pred)

    mse_ = performance.mse(y_pred, y_true)
    assert mse_ == 5.125/6
    mse_ = performance.mse(y_pred, y_true, ts_weights=[0, 1])
    assert mse_ == 1
    mse_ = performance.mse(y_pred, y_true, timestamps_weights=[1, 0, 1])
    assert mse_ == 8.25/8
    mse_ = performance.mse(y_pred, y_true, multioutput=[0, 1])
    assert mse_ == 1/2
    mse_ = performance.mse(y_pred, y_true, multioutput="raw_values")
    np.testing.assert_almost_equal(mse_, [7.25/6, 1/2])
    mse_ = performance.mse(
        y_pred,
        y_true,
        ts_weights=[0, 1],
        timestamps_weights=[0, 1, 0]
    )
    assert mse_ == 1
    mse_ = performance.mse(
        y_pred,
        y_true,
        ts_weights=[0, 1],
        timestamps_weights=[0, 1, 0],
        multioutput=[0, 1]
    )
    assert mse_ == 1
    mse_ = performance.mse(
        y_pred,
        y_true,
        ts_weights=[0, 1],
        timestamps_weights=[0, 1, 0],
        multioutput="raw_values"
    )
    np.testing.assert_almost_equal(mse_, [1, 1])

    rng = np.random.default_rng(0)
    sz = 1000
    y_true = [0] * sz

    # Normal distribution
    y_pred = np.array(y_true) + rng.normal(0, 0.5, sz)
    mse_ = performance.mse(y_pred, y_true)
    np.testing.assert_almost_equal(mse_, 0.25, decimal=2)

    # Uniform distribution
    y_pred = np.array(y_true) + rng.uniform(-0.5, 0.5, sz)
    mse_ = performance.mse(y_pred, y_true)
    np.testing.assert_almost_equal(mse_, 1/12, decimal=2)


def test_mase():

    y_train= [[[3, 4], [5, 5], [5, 6], [6, 7], [7, 8]]]
    y_true = [
        [[1, 2], [2, 3], [3, 4]],
    ]
    y_pred = [
        [[0, 1], [1, 2], [2, 3]]
    ]

    assert performance.mse(y_pred, y_true) == performance.mae(y_true, y_pred)

    mase_ = performance.mase(y_pred, y_true, y_train)
    assert mase_ == 1
    mase_ = performance.mase(y_pred*100, y_true*100, y_train*100)
    assert mase_ == 1

    y_train = [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]
    mase_ = performance.mase(y_pred, y_true, y_train)
    assert mase_ == 9/13
    mase_ = performance.mase(y_pred, y_true, y_train, seasonal_period=5)
    assert mase_ == 1
    mase_ = performance.mase(y_pred, y_true, y_train, seasonal_period=5, multioutput=[0, 1])
    assert mase_ == 1
    mase_ = performance.mase(y_pred, y_true, y_train, seasonal_period=1, multioutput="raw_values")
    np.testing.assert_allclose(mase_, [9/13, 9/13])

    y_train = [0] * 10
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        mase_ = performance.mase(y_pred, y_true, y_train)
    assert mase_ == np.inf
