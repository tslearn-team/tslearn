"""
The :mod:`tslearn.datasets` module provides simplified access to standard time
series datasets.
"""

import zipfile
import tempfile
import shutil
import os
import warnings
from urllib.request import urlretrieve

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def extract_from_zip_url(url, target_dir=None, verbose=False):
    """Download a zip file from its URL and unzip it.

    A `RuntimeWarning` is printed on failure.

    Parameters
    ----------
    url : string
        URL from which to download.
    target_dir : str or None (default: None)
        Directory to be used to extract unzipped downloaded files.
    verbose : bool (default: False)
        Whether to print information about the process (cached files used, ...)

    Returns
    -------
    str or None
        Directory in which the zip file has been extracted if the process was
        successful, None otherwise
    """
    fname = os.path.basename(url)
    tmpdir = tempfile.mkdtemp()
    local_zip_fname = os.path.join(tmpdir, fname)
    urlretrieve(url, local_zip_fname)
    os.makedirs(target_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(local_zip_fname, "r") as f:
            f.extractall(path=target_dir)
        if verbose:
            print("Successfully extracted file %s to path %s" %
                  (local_zip_fname, target_dir))
        return target_dir
    except zipfile.BadZipFile:
        warnings.warn("Corrupted or missing zip file encountered, aborting",
                      category=RuntimeWarning)
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def in_file_string_replace(filename, old_string, new_string):
    """String replacement within a text file. It is used to fix typos in
    downloaded csv file.

    The code was modified from "https://stackoverflow.com/questions/4128144/"

    Parameters
    ----------
    filename : str
        Path to the file where strings should be replaced
    old_string : str
        The string to be replaced in the file.
    new_string : str
        The new string that will replace old_string
    """
    with open(filename) as f:
        s = f.read()

    with open(filename, 'w') as f:
        s = s.replace(old_string, new_string)
        f.write(s)
