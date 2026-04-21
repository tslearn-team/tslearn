r"""
The :mod:`tslearn.datasets` module provides simplified access to standard time
series datasets.

.. note::
    **MacOS users:** If you encounter SSL certificate errors when downloading
    datasets, you may need to install certificates for your Python
    installation. Run the following command::

        /Applications/Python<VERSION>/Install\ Certificates.command

    Alternatively, install the ``certifi`` package::

        pip install certifi
"""

from .datasets import extract_from_zip_url, in_file_string_replace
from .ucr_uea import UCR_UEA_datasets
from .cached import CachedDatasets

__all__ = [
    "extract_from_zip_url", "in_file_string_replace",
    "UCR_UEA_datasets",
    "CachedDatasets"
]
