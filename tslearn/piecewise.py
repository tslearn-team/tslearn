import numpy
from scipy.stats import norm
from sklearn.base import TransformerMixin

from tslearn.utils import npy3d_time_series_dataset, npy2d_time_series
from tslearn.cysax import distance_sax


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class PiecewiseAggregateApproximation(TransformerMixin):
    """Piecewise Aggregate Approximation (PAA) transformation as defined in [1]_.

    Parameters
    ----------
    n_segments : int
        Number of PAA segments to compute

    References
    ----------
    .. [1] E. Keogh & M. Pazzani. Scaling up dynamic time warping for datamining applications. SIGKDD 2000
       (pp. 285-289).
    """
    def __init__(self, n_segments):
        self.n_segments = n_segments
        self.size_fitted_ = -1

    def _fit(self, X, y=None):
        self.size_fitted_ = X.shape[1]
        return self

    def fit(self, X, y=None):
        """Fit a PAA representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        PiecewiseAggregateApproximation
            self
        """
        X_ = npy3d_time_series_dataset(X)
        return self._fit(X_, y)

    def _transform(self, X, y=None):
        n_ts, sz, d = X.shape
        X_transformed = numpy.empty((n_ts, self.n_segments, d))
        sz_segment = sz // self.n_segments
        for i_seg in range(self.n_segments):
            start = i_seg * sz_segment
            end = start + sz_segment
            X_transformed[:, i_seg, :] = X[:, start:end, :].mean(axis=1)
        return X_transformed

    def transform(self, X, y=None):
        """Transform a dataset of time series into its PAA representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, n_segments, d)
            PAA-Transformed dataset
        """
        X_ = npy3d_time_series_dataset(X)
        return self._transform(X_, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit a PAA representation and transform the data accordingly.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, n_segments, d)
            PAA-Transformed dataset
        """
        X_ = npy3d_time_series_dataset(X)
        return self._fit(X_)._transform(X_)

    def distance_paa(self, paa1, paa2):
        """Compute distance between PAA representations as defined in [1]_.

        Parameters
        ----------
        paa1 : array-like
            PAA representation of a time series
        paa2 : array-like
            PAA representation of another time series

        Returns
        -------
        float
            PAA distance
        """
        if self.size_fitted_ < 0:
            raise ValueError("Model not fitted yet: cannot be used for distance computation.")
        else:
            return numpy.linalg.norm(paa1 - paa2) * numpy.sqrt(self.size_fitted_ / self.n_segments)

    def distance(self, ts1, ts2):
        """Compute distance between PAA representations as defined in [1]_.

        Parameters
        ----------
        ts1 : array-like
            A time series
        ts2 : array-like
            Another time series

        Returns
        -------
        float
            PAA distance
        """
        paa = self.transform([ts1, ts2])
        return self.distance_paa(paa[0], paa[1])


class SymbolicAggregateApproximation(PiecewiseAggregateApproximation):
    """Symbolic Aggregate approXimation (SAX) transformation as defined in [2]_.

    Parameters
    ----------
    n_segments : int
        Number of PAA segments to compute
    alphabet_size : int
        Number of SAX symbols to use

    Attributes
    ----------
    breakpoints_ : numpy.ndarray of shape (alphabet_size - 1, )
        List of breakpoints used to generate SAX symbols

    References
    ----------
    .. [2] J. Lin, E. Keogh, L. Wei, et al. Experiencing SAX: a novel symbolic representation of time series.
       Data Mining and Knowledge Discovery, 2007. vol. 15(107)
    """
    def __init__(self, n_segments, alphabet_size):
        PiecewiseAggregateApproximation.__init__(self, n_segments)
        self.alphabet_size = alphabet_size
        self.breakpoints_ = norm.ppf([float(a) / self.alphabet_size for a in range(1, self.alphabet_size)])

    def fit(self, X, y=None):
        """Fit a SAX representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        SymbolicAggregateApproximation
            self
        """
        return PiecewiseAggregateApproximation.fit(self, X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit a SAX representation and transform the data accordingly.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of integers with shape (n_ts, n_segments, d)
            SAX-Transformed dataset
        """
        X_ = npy3d_time_series_dataset(X)
        return self._fit(X_)._transform(X_)

    def _transform(self, X, y=None):
        X_paa = PiecewiseAggregateApproximation._transform(self, X, y)
        X_sax = numpy.zeros(X_paa.shape, dtype=numpy.int) - 1
        for idx_bp, bp in enumerate(self.breakpoints_):
            indices = numpy.logical_and(X_sax < 0, X_paa < bp)
            X_sax[indices] = idx_bp
        X_sax[X_sax < 0] = self.alphabet_size - 1
        return X_sax

    def transform(self, X, y=None):
        """Transform a dataset of time series into its SAX representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of integers with shape (n_ts, n_segments, d)
            SAX-Transformed dataset
        """
        X_ = npy3d_time_series_dataset(X)
        return self._transform(X_, y)

    def distance_sax(self, sax1, sax2):
        """Compute distance between SAX representations as defined in [2]_.

        Parameters
        ----------
        sax1 : array-like
            SAX representation of a time series
        sax2 : array-like
            SAX representation of another time series

        Returns
        -------
        float
            SAX distance
        """
        if self.size_fitted_ < 0:
            raise ValueError("Model not fitted yet: cannot be used for distance computation.")
        else:
            return distance_sax(sax1, sax2, self.breakpoints_, self.size_fitted_)

    def distance(self, ts1, ts2):
        """Compute distance between SAX representations as defined in [2]_.

        Parameters
        ----------
        ts1 : array-like
            A time series
        ts2 : array-like
            Another time series

        Returns
        -------
        float
            SAX distance
        """
        sax = self.transform([ts1, ts2])
        return self.distance_sax(sax[0], sax[1])

