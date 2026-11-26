from pathlib import Path

import numpy as np
import pytest

from tslearn.datasets.ucr_uea import UCR_UEA_datasets
from tslearn.datasets.cached import CachedDatasets


def test_ucr_uea_datasets(require_ucr_datasets):
    data_loader = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = data_loader.load_dataset("Trace")
    assert X_train.shape == (100, 275, 1)
    assert y_train.shape == (100,)
    assert X_test.shape == (100, 275, 1)
    assert y_test.shape == (100,)


def test_cached_datasets():
    data_loader = CachedDatasets()
    cached = data_loader.list_datasets()
    assert "Trace" in cached


def test_root_dir(require_ucr_datasets, tmp_path):
    """Check that datasets are cached in the specified root directory."""
    # Make sure the Trace dataset is cached in the default location
    data_loader = UCR_UEA_datasets()
    _ = data_loader.load_dataset("Trace")

    # Check that the cached directory of a new data loader is empty when
    # using a custom location, and that loading Trace populates it
    data_loader = UCR_UEA_datasets(root_dir=tmp_path)
    cached_dir = data_loader.list_cached_datasets()
    assert len(cached_dir) == 0

    data_loader.load_dataset("Trace")
    cached_dir = data_loader.list_cached_datasets()
    assert "Trace" in cached_dir

    assert (Path(tmp_path) / "UCR_UEA" / "Trace").exists()


def test_default_data_home(require_ucr_datasets, monkeypatch, tmp_path):
    """Check that datasets are cached in the specified root directory."""
    # Make sure the Trace dataset is cached in the default location
    data_loader = UCR_UEA_datasets()
    _ = data_loader.load_dataset("Trace")

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    # Check that the cached directory of a new data loader is empty when
    # using a custom location, and that loading Trace populates it
    data_loader = UCR_UEA_datasets()
    assert data_loader._data_dir == Path(tmp_path) / "tslearn" / "UCR_UEA"
    cached_dir = data_loader.list_cached_datasets()
    assert len(cached_dir) == 0


def test_deprecated_fatal_ecg_name(tmp_path):
    """Check that the misspelled NonInvasiveFatalECG names load the Fetal ones."""
    name = "NonInvasiveFetalECGThorax1"
    data_dir = tmp_path / "UCR_UEA" / name
    data_dir.mkdir(parents=True)
    for split in ["TRAIN", "TEST"]:
        np.savetxt(data_dir / f"{name}_{split}.txt", [[1, 0., 1.], [2, 1., 0.]])

    data_loader = UCR_UEA_datasets(root_dir=tmp_path)
    with pytest.warns(DeprecationWarning, match=name):
        X_train, y_train, _, _ = data_loader.load_dataset(
            "NonInvasiveFatalECGThorax1"
        )
    assert X_train.shape == (2, 2, 1)
    assert y_train.tolist() == [1, 2]


def test_fetal_ecg_name_listed(require_ucr_datasets):
    datasets = UCR_UEA_datasets().list_univariate_datasets()
    assert "NonInvasiveFetalECGThorax1" in datasets
    assert "NonInvasiveFatalECGThorax1" not in datasets


def test_load_dataset_replaces_existing_dataset_dir(tmp_path, monkeypatch):
    """A leftover dataset directory is cleared before downloading again.

    ``Path.unlink`` raises ``PermissionError`` (not ``IsADirectoryError``)
    on directories on macOS and Windows, which used to make ``load_dataset``
    crash there instead of re-downloading.
    """
    import warnings

    from tslearn.datasets import ucr_uea

    downloads = []

    def fake_extract(url, target_dir=None, verbose=False):
        downloads.append(target_dir)
        assert not Path(target_dir).exists()
        return None

    monkeypatch.setattr(ucr_uea, "extract_from_zip_url", fake_extract)
    data_loader = UCR_UEA_datasets(use_cache=False, root_dir=tmp_path)
    dataset_dir = Path(tmp_path) / "UCR_UEA" / "Trace"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "leftover.txt").write_text("partial download")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert data_loader.load_dataset("Trace") == (None, None, None, None)
    assert downloads == [dataset_dir]
