from tempfile import gettempdir
from os.path import join

import numpy as np
import pickle
from numpy.testing import assert_allclose

import pytest

import tslearn.utils

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


EXAMPLE_FILE = join(gettempdir(), "tslearn_pytest_file.txt")


def test_save_load_random():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    dataset = rng.randn(n, sz, d)
    tslearn.utils.save_timeseries_txt(EXAMPLE_FILE, dataset)
    reloaded_dataset = tslearn.utils.load_timeseries_txt(EXAMPLE_FILE)
    assert_allclose(dataset, reloaded_dataset)


def test_save_load_known():
    dataset = tslearn.utils.to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3]])
    tslearn.utils.save_timeseries_txt(EXAMPLE_FILE, dataset)
    reloaded_dataset = tslearn.utils.load_timeseries_txt(EXAMPLE_FILE)
    for ts0, ts1 in zip(dataset, reloaded_dataset):
        assert_allclose(ts0[:tslearn.utils.ts_size(ts0)],
                        ts1[:tslearn.utils.ts_size(ts1)])


def test_label_categorizer():
    y = np.array([-1, 2, 1, 1, 2])
    lc = tslearn.utils.LabelCategorizer()
    lc.fit(y)

    s = pickle.dumps(lc)
    lc2 = pickle.loads(s)
    y_tr = lc2.inverse_transform([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    ref = np.array([1., 2., -1.])

    assert_allclose(ref, y_tr)


def test_conversions():
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    tslearn_dataset = rng.randn(n, sz, d)

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_pyts_dataset(
            tslearn.utils.to_pyts_dataset(tslearn_dataset)
        )
    )

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_seglearn_dataset(
            tslearn.utils.to_seglearn_dataset(tslearn_dataset)
        )
    )

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_stumpy_dataset(
            tslearn.utils.to_stumpy_dataset(tslearn_dataset)
        )
    )


def test_conversions_with_pandas():
    pytest.importorskip('pandas')
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    tslearn_dataset = rng.randn(n, sz, d)

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_sktime_dataset(
            tslearn.utils.to_sktime_dataset(tslearn_dataset)
        )
    )

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_tsfresh_dataset(
            tslearn.utils.to_tsfresh_dataset(tslearn_dataset)
        )
    )

    tslearn_dataset = rng.randn(1, sz, d)
    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_pyflux_dataset(
            tslearn.utils.to_pyflux_dataset(tslearn_dataset)
        )
    )


def test_conversions_cesium():
    pytest.importorskip('cesium')
    n, sz, d = 15, 10, 3
    rng = np.random.RandomState(0)
    tslearn_dataset = rng.randn(n, sz, d)

    assert_allclose(
        tslearn_dataset,
        tslearn.utils.from_cesium_dataset(
            tslearn.utils.to_cesium_dataset(tslearn_dataset)
        )
    )
