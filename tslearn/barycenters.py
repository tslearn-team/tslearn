from tslearn.utils import npy3d_time_series_dataset


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class EuclideanBarycenter:
    """Standard Euclidean barycenter computed from a set of time series."""
    def fit(self, X):
        """Compute the barycenter from a dataset of time series.

        Parameters
        ----------
        X : array-like, shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        numpy.array of shape (sz, d)
            Barycenter of the provided time series dataset.
        """
        X_ = npy3d_time_series_dataset(X)
        return X_.mean(axis=0)