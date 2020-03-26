import pytest


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
