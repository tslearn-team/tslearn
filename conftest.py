import warnings

import pytest

from tslearn.datasets import UCR_UEA_datasets


def pytest_collection_modifyitems(config, items):
    try:
        import pandas
    except ImportError:
        pandas = None

    try:
        import cesium
    except ImportError:
        cesium = None

    if pandas is None:
        skip_marker = pytest.mark.skip(reason="pandas not installed!")
        for item in items:
            if item.name in [
                "tslearn.utils.from_tsfresh_dataset",
                "tslearn.utils.to_tsfresh_dataset",
                "tslearn.utils.from_sktime_dataset",
                "tslearn.utils.to_sktime_dataset",
                "tslearn.utils.from_pyflux_dataset",
                "tslearn.utils.to_pyflux_dataset",
                "tslearn.utils.from_cesium_dataset",
                "tslearn.utils.to_cesium_dataset",
            ]:
                item.add_marker(skip_marker)
    if cesium is None:
        skip_marker = pytest.mark.skip(reason="cesium not installed!")
        for item in items:
            if item.name in [
                "tslearn.utils.to_cesium_dataset",
                "tslearn.utils.from_cesium_dataset",
            ]:
                item.add_marker(skip_marker)

    # Skip related doctests if UCR UEA datasets cannot be fetched
    ucr_uea_datasets = True
    try:
        datasets = UCR_UEA_datasets()
        datasets.cache_all()
        ucr_uea_datasets = bool(datasets.list_datasets())
    except Exception as exc:
        warnings.warn("Error caching UCR UEA datasets: {}".format(exc))

    if not ucr_uea_datasets:
        warnings.warn("Skipping doctests requiring UCR UEA dataset download")
        skip_marker = pytest.mark.skip(reason="Datasets not cached!")
        for item in items:
            if item.name in [
                "tslearn.datasets.ucr_uea.UCR_UEA_datasets.list_datasets",
                "tslearn.datasets.ucr_uea.UCR_UEA_datasets.list_multivariate_datasets",
                "tslearn.datasets.ucr_uea.UCR_UEA_datasets.list_univariate_datasets",
                "tslearn.datasets.ucr_uea.UCR_UEA_datasets.load_dataset",
            ]:
                item.add_marker(skip_marker)
