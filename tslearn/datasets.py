"""
The :mod:`tslearn.datasets` module provides simplified access to standard time series datasets.
"""

import numpy
import zipfile
import tempfile
import shutil
import os
import sys
import csv
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
try:
    from zipfile import BadZipFile as BadZipFile
except ImportError:
    from zipfile import BadZipfile as BadZipFile

from tslearn.utils import to_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def extract_from_zip_url(url, target_dir=None, verbose=False):
    """Download a zip file from its URL and unzip it.

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
        Directory in which the zip file has been extracted if the process was successful, None otherwise
    """
    fname = os.path.basename(url)
    tmpdir = tempfile.mkdtemp()
    local_zip_fname = os.path.join(tmpdir, fname)
    urlretrieve(url, local_zip_fname)
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        zipfile.ZipFile(local_zip_fname, "r").extractall(path=target_dir)
        shutil.rmtree(tmpdir)
        if verbose:
            print("Successfully extracted file %s to path %s" % (local_zip_fname, target_dir))
        return target_dir
    except BadZipFile:
        shutil.rmtree(tmpdir)
        if verbose:
            sys.stderr.write("Corrupted zip file encountered, aborting.\n")
        return None


class UCR_UEA_datasets(object):
    """A convenience class to access UCR/UEA time series datasets.

    When using one (or several) of these datasets in research projects, please cite [1]_.

    Parameters
    ----------
    use_cache : bool (default: True)
        Whether a cached version of the dataset should be used, if found.

    Note
    ----
        Downloading dataset files can be time-consuming, it is recommended using `use_cache=True` (default) in order to
        only experience downloading time once per dataset and work on a cached version of the datasets after it.

    References
    ----------
    .. [1] A. Bagnall, J. Lines, W. Vickers and E. Keogh, The UEA & UCR Time Series Classification Repository,
       www.timeseriesclassification.com
    """
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        base_dir = os.path.expanduser(os.path.join("~", ".tslearn", "datasets", "UCR_UEA"))
        self._data_dir = base_dir
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)
        try:
            url_baseline = "http://www.timeseriesclassification.com/singleTrainTest.csv"
            self._baseline_scores_filename = os.path.join(self._data_dir, os.path.basename(url_baseline))
            urlretrieve(url_baseline, self._baseline_scores_filename)
        except:
            self._baseline_scores_filename = None

        self._ignore_list = ["Data Descriptions"]
        # File names for datasets for which it is not obvious
        self._filenames = {"CinCECGtorso": "CinC_ECG_torso",
                           "CricketX": "Cricket_X",
                           "CricketY": "Cricket_Y",
                           "CricketZ": "Cricket_Z",
                           "FiftyWords": "50words",
                           "Lightning2": "Lighting2",
                           "Lightning7": "Lighting7",
                           "NonInvasiveFatalECGThorax1": "NonInvasiveFetalECG_Thorax1",
                           "NonInvasiveFatalECGThorax2": "NonInvasiveFetalECG_Thorax2",
                           "GunPoint": "Gun_Point",
                           "SonyAIBORobotSurface1": "SonyAIBORobotSurface",
                           "SonyAIBORobotSurface2": "SonyAIBORobotSurfaceII",
                           "SyntheticControl": "synthetic_control",
                           "TwoPatterns": "Two_Patterns",
                           "UWaveGestureLibraryX": "UWaveGestureLibrary_X",
                           "UWaveGestureLibraryY": "UWaveGestureLibrary_Y",
                           "UWaveGestureLibraryZ": "UWaveGestureLibrary_Z",
                           "WordSynonyms": "WordsSynonyms"}

    def baseline_accuracy(self, list_datasets=None, list_methods=None):
        """Report baseline performances as provided by UEA/UCR website.

        Parameters
        ----------
        list_datasets: list or None (default: None)
            A list of strings indicating for which datasets performance should be reported.
            If None, performance is reported for all datasets.
        list_methods: list or None (default: None)
            A list of baselines methods for which performance should be reported.
            If None, performance for all baseline methods is reported.

        Returns
        -------
        dict
            A dictionary in which keys are dataset names and associated values are themselves
            dictionaries that provide accuracy scores for the requested methods.

        Examples
        --------
        >>> uea_ucr = UCR_UEA_datasets()
        >>> dict_acc = uea_ucr.baseline_accuracy(list_datasets=["Adiac", "ChlorineConcentration"], \
                                                 list_methods=["C45"])
        >>> len(dict_acc)
        2
        >>> dict_acc["Adiac"]  # doctest: +ELLIPSIS
        {'C45': 0.542199...}
        >>> dict_acc = uea_ucr.baseline_accuracy()
        >>> len(dict_acc)
        85
        """
        d_out = {}
        for perfs_dict in csv.DictReader(open(self._baseline_scores_filename, "r"), delimiter=","):
            dataset_name = perfs_dict[""]
            if list_datasets is None or dataset_name in list_datasets:
                d_out[dataset_name] = {}
                for m in perfs_dict.keys():
                    if m != "" and (list_methods is None or m in list_methods):
                        try:
                            d_out[dataset_name][m] = float(perfs_dict[m])
                        except ValueError:  # Missing score case (score is then "")
                            pass
        return d_out

    def list_datasets(self):
        """List datasets in the UCR/UEA archive.

        Examples
        --------
        >>> l = UCR_UEA_datasets().list_datasets()
        >>> len(l)
        85
        """
        datasets = []
        for perfs_dict in csv.DictReader(open(self._baseline_scores_filename, "r"), delimiter=","):
            datasets.append(perfs_dict[""])
        return datasets

    def list_cached_datasets(self):
        """List datasets from the UCR/UEA archive that are avilable in cache.

        Examples
        --------
        >>> l = UCR_UEA_datasets().list_cached_datasets()
        >>> len(l) >= 0 and len(l) <= len(UCR_UEA_datasets().list_datasets())
        True
        """
        return [path for path in os.listdir(self._data_dir)
                if os.path.isdir(os.path.join(self._data_dir, path)) and path not in self._ignore_list]

    def load_dataset(self, dataset_name):
        """Load a dataset from the UCR/UEA archive from its name.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset. Should be in the list returned by `list_datasets`

        Returns
        -------
        numpy.ndarray of shape (n_ts_train, sz, d) or None
            Training time series. None if unsuccessful.
        numpy.ndarray of integers with shape (n_ts_train, ) or None
            Training labels. None if unsuccessful.
        numpy.ndarray of shape (n_ts_test, sz, d) or None
            Test time series. None if unsuccessful.
        numpy.ndarray of integers with shape (n_ts_test, ) or None
            Test labels. None if unsuccessful.

        Examples
        --------
        >>> X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("TwoPatterns")
        >>> print(X_train.shape)
        (1000, 128, 1)
        >>> print(y_train.shape)
        (1000,)
        >>> X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("DatasetThatDoesNotExist")
        >>> print(X_train)
        None
        """
        full_path = os.path.join(self._data_dir, dataset_name)
        fname_train = self._filenames.get(dataset_name, dataset_name) + "_TRAIN.txt"
        fname_test = self._filenames.get(dataset_name, dataset_name) + "_TEST.txt"
        if not os.path.exists(os.path.join(full_path, fname_train)) or \
            not os.path.exists(os.path.join(full_path, fname_test)):
            url = "http://www.timeseriesclassification.com/Downloads/%s.zip" % dataset_name
            for fname in [fname_train, fname_test]:
                if os.path.exists(os.path.join(full_path, fname)):
                    os.remove(os.path.join(full_path, fname))
            extract_from_zip_url(url, target_dir=self._data_dir, verbose=False)
        try:
            data_train = numpy.loadtxt(os.path.join(full_path, fname_train), delimiter=",")
            data_test = numpy.loadtxt(os.path.join(full_path, fname_test), delimiter=",")
        except:
            return None, None, None, None
        X_train = to_time_series_dataset(data_train[:, 1:])
        y_train = data_train[:, 0].astype(numpy.int)
        X_test = to_time_series_dataset(data_test[:, 1:])
        y_test = data_test[:, 0].astype(numpy.int)
        return X_train, y_train, X_test, y_test

    def cache_all(self):
        """Cache all datasets from the UCR/UEA archive for later use.
        """
        for dataset_name in self.list_datasets():
            try:
                self.load_dataset(dataset_name)
            except:
                sys.stderr.write("Could not cache dataset %s properly.\n" % dataset_name)


class CachedDatasets(object):
    """A convenience class to access cached time series datasets.

    When using the Trace dataset, please cite [1]_.

    References
    ----------
    .. [1] A. Bagnall, J. Lines, W. Vickers and E. Keogh, The UEA & UCR Time Series Classification Repository,
       www.timeseriesclassification.com
    """
    def __init__(self):
        self.path = os.path.join(os.path.dirname(__file__), ".cached_datasets")

    def list_datasets(self):
        """List cached datasets."""
        return [fname[:fname.rfind(".")]
                for fname in os.listdir(self.path)
                if fname.endswith(".npz")]

    def load_dataset(self, dataset_name):
        """Load a cached dataset from its name.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset. Should be in the list returned by `list_datasets`

        Returns
        -------
        numpy.ndarray of shape (n_ts_train, sz, d) or None
            Training time series. None if unsuccessful.
        numpy.ndarray of integers with shape (n_ts_train, ) or None
            Training labels. None if unsuccessful.
        numpy.ndarray of shape (n_ts_test, sz, d) or None
            Test time series. None if unsuccessful.
        numpy.ndarray of integers with shape (n_ts_test, ) or None
            Test labels. None if unsuccessful.

        Examples
        --------
        >>> X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
        >>> print(X_train.shape)
        (100, 275, 1)
        >>> print(y_train.shape)
        (100,)
        """
        npzfile = numpy.load(os.path.join(self.path, dataset_name + ".npz"))
        X_train = npzfile["X_train"]
        X_test = npzfile["X_test"]
        y_train = npzfile["y_train"]
        y_test = npzfile["y_test"]
        return X_train, y_train, X_test, y_test
