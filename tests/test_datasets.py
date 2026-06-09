from pathlib import Path

from tslearn.datasets.ucr_uea import UCR_UEA_datasets
from tslearn.datasets.cached import CachedDatasets


def test_ucr_uea_datasets():
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


def test_root_dir(tmp_path):
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

