import numpy
import os

class CachedDatasets:
    """A convenience class to access cached time series datasets.

    Note, that these *cached datasets* are statically included into *tslearn*
    and are distinct from the ones in :class:`UCR_UEA_datasets`.

    When using the Trace dataset, please cite [1]_.

    See Also
    --------
    UCR_UEA_datasets : Provides more datasets and supports caching.

    References
    ----------
    .. [1] A. Bagnall, J. Lines, W. Vickers and E. Keogh, The UEA & UCR Time
       Series Classification Repository, www.timeseriesclassification.com
    """
    def __init__(self):
        self.path = os.path.join(os.path.dirname(__file__),
                                 "..",
                                 ".cached_datasets")

    def list_datasets(self):
        """List cached datasets.

        Examples
        --------
        >>> from tslearn.datasets import UCR_UEA_datasets
        >>> _ = UCR_UEA_datasets().load_dataset("Trace")
        >>> cached = UCR_UEA_datasets().list_cached_datasets()
        >>> "Trace" in cached
        True

        Returns
        -------
        list of str:
            A list of names of all cached (univariate and multivariate) dataset
            namas.
        """
        return [fname[:fname.rfind(".")]
                for fname in os.listdir(self.path)
                if fname.endswith(".npz")]

    def load_dataset(self, dataset_name):
        """Load a cached dataset from its name.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset. Should be in the list returned by
            :meth:`~list_datasets`.

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
        >>> data_loader = CachedDatasets()
        >>> X_train, y_train, X_test, y_test = data_loader.load_dataset(
        ...                                        "Trace")
        >>> print(X_train.shape)
        (100, 275, 1)
        >>> print(y_train.shape)
        (100,)

        Raises
        ------
        IOError
            If the dataset does not exist or cannot be read.
        """
        npzfile = numpy.load(os.path.join(self.path, dataset_name + ".npz"))
        X_train = npzfile["X_train"]
        X_test = npzfile["X_test"]
        y_train = npzfile["y_train"]
        y_test = npzfile["y_test"]
        return X_train, y_train, X_test, y_test
