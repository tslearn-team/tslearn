"""
The :mod:`tslearn.datasets` module provides simplified access to standard time
series datasets.
"""

from .datasets import extract_from_zip_url, in_file_string_replace
from .ucr_uea import UCR_UEA_datasets
from .cached import CachedDatasets

__all__ = [
    "extract_from_zip_url", "in_file_string_replace",
    "UCR_UEA_datasets",
    "CachedDatasets"
]