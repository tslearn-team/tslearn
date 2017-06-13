import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d

from tslearn.metrics import dtw_path, lr_dtw_path


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class DTWSampler(BaseEstimator, TransformerMixin):
    """A class for non-linear DTW-based resampling of time series as described in [1]_.

    The principle is to use a modality (or a set of modalities) to perform DTW alignment with respect to a reference
    and then resample other modalities using the obtained DTW path.
    Note that LR-DTW algorithm can also be used as a substitute for DTW if necessary.

    A typical usage should be:

    1. build the sampler by calling the constructor
    2. fit (i.e. provide reference time series)
    3. call prepare_transform to perform DTW between base modalities of the targets and those of the reference.
    4. call transform to get resampled time series for all other modalities

    If one wants to use LR-DTW instead of DTW at the core of this method, the metric attribute should be set to
    "lrdtw".

    Parameters
    ----------
    n_samples : int (default: 100)
        Size of generated time series.
    interp_kind : str (default: "slinear")
        Interpolation kind to be used in the call to ``scipy.interpolate.interp1d``.
    metric : {"dtw", "lrdtw"} (default: "dtw")
        Metric to be used for time series alignment.
    gamma_lr_dtw : float (default: 1.)
        Gamma parameter for LR-DTW (only used if metric="lrdtw").
    
    References
    ----------
    .. [1] R. Dupas et al. Identifying seasonal patterns of phosphorus storm dynamics with dynamic time warping.
       Water Resources Research, vol. 51 (11), pp. 8868--8882, 2015.
    """
    def __init__(self, n_samples=100, interp_kind="slinear", metric="dtw", gamma_lr_dtw=1.):
        self.n_samples = n_samples
        self.interp_kind = interp_kind
        self.reference_series_ = None

        self.saved_dtw_paths_ = None

        self.gamma_lr_dtw = gamma_lr_dtw

        self.metric = metric

    def fit(self, X):
        """Register X as the reference series and interpolate it to get a series of size nsamples.

        Parameters
        ----------
        X : numpy.ndarray
            A time series.

        Returns
        -------
        DTWSampler
            self
        """
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        elif X.ndim == 2:
            if X.shape[0] == 1:
                X = X.reshape((-1, 1))
        elif X.ndim == 3 and X.shape[0] == 1:
            X = X.reshape((-1, 1))
        else:
            raise ValueError("dimension mismatch")

        end = first_non_finite_index(X)
        self.reference_series_ = _resampled(X[:end], n_samples=self.n_samples, kind=self.interp_kind)
        return self

    def prepare_transform(self, ts_to_be_rescaled):
        """Prepare the model for temporal resampling by computing DTW alignment path between the reference time series
        and a time series to be rescaled or a set of time series to be rescaled.
        
        If ts_to_be_rescaled contains a single time series, all series from the dataset will be rescaled using the
        DTW path between that time series and the reference one, otherwise, the X array given at transform time
        should have the same number of time series (X.shape[0]) as ts_to_be_rescaled.

        Parameters
        ----------
        ts_to_be_rescaled : numpy.ndarray
            A time series dataset of base modalities of shape (n_ts, sz, d) with
            ``d = self.reference_series_.shape[-1]``
        """
        if ts_to_be_rescaled.ndim == 1:
            ts_to_be_rescaled = ts_to_be_rescaled.reshape((1, -1, 1))
        elif ts_to_be_rescaled.ndim == 2:
            if ts_to_be_rescaled.shape[1] == self.reference_series_.shape[1]:
                ts_to_be_rescaled = ts_to_be_rescaled.reshape((1, -1, ts_to_be_rescaled.shape[1]))
            else:
                ts_to_be_rescaled = ts_to_be_rescaled.reshape((ts_to_be_rescaled.shape[0], -1, 1))
        elif ts_to_be_rescaled.ndim >= 4:
            raise ValueError
        # Now ts_to_be_rescaled is of shape n_ts, sz, d 
        # with d = self.reference_series.shape[-1]
        self.saved_dtw_paths_ = []
        for ts in ts_to_be_rescaled:
            end = first_non_finite_index(ts)
            resampled_ts = _resampled(ts[:end], n_samples=self.n_samples, kind=self.interp_kind)
            if self.metric == "dtw":
                path, d = dtw_path(self.reference_series_, resampled_ts)
            elif self.metric == "lrdtw":
                path, d = lr_dtw_path(self.reference_series_, resampled_ts, gamma=self.gamma_lr_dtw)
            else:
                raise ValueError("Unknown alignment function")
            self.saved_dtw_paths_.append(path)

    def transform(self, X):
        """Resample time series from dataset X according to resampling computed at the prepare_transform stage.

        Parameters
        ----------
        X : numpy.ndarray
            A time series dataset to be resampled (3-dimensional array).

        Returns
        -------
        numpy.ndarray
            The transformed time series dataset
        """
        assert X.shape[0] == len(self.saved_dtw_paths_) or len(self.saved_dtw_paths_) == 1
        X_resampled = numpy.zeros((X.shape[0], self.n_samples, X.shape[2]))
        for i in range(X.shape[0]):
            end = first_non_finite_index(X[i])
            X_resampled[i] = _resampled(X[i, :end], n_samples=self.n_samples, kind=self.interp_kind)
            
            indices_xy = [[] for _ in range(self.n_samples)]

            if len(self.saved_dtw_paths_) == 1:
                path = self.saved_dtw_paths_[0]
            else:
                path = self.saved_dtw_paths_[i]

            if self.metric == "dtw":
                for t_ref, t_current in path:
                    indices_xy[t_ref].append(t_current)
                for j in range(X.shape[2]):
                    ynew = numpy.array([numpy.mean(X_resampled[i, indices, j]) for indices in indices_xy])
                    X_resampled[i, :, j] = ynew
            elif self.metric == "lrdtw":
                ynew = numpy.empty((self.n_samples, X.shape[2]))
                for t in range(self.n_samples):
                    weights = path[t] / path[t].sum()
                    ynew[t] = numpy.sum(X_resampled[i] * weights.reshape((-1, 1)), axis=0)
                X_resampled[i] = ynew
            else:
                raise ValueError("Unknown alignment function")
        return X_resampled


def _resampled(X, n_samples=100, kind="slinear"):
    """Perform resampling for time series X using the method given in kind.

    Examples
    --------
    >>> _resampled(numpy.array([[0], [1]]), n_samples=5) # doctest: +NORMALIZE_WHITESPACE
    array([[ 0. ],
           [ 0.25],
           [ 0.5 ],
           [ 0.75],
           [ 1. ]])
    """
    if X.ndim == 1:
        X = X.reshape((-1, 1))
    assert X.ndim == 2
    X_out = numpy.zeros((n_samples, X.shape[-1]))
    xnew = numpy.linspace(0, 1, n_samples)
    for di in range(X.shape[-1]):
        f = interp1d(numpy.linspace(0, 1, X.shape[0]), X[:, di], kind=kind)
        X_out[:, di] = f(xnew)
    return X_out


def first_non_finite_index(X):
    """Return first index at which time series in X is non-finite.

    Parameters
    ----------
    X : numpy.ndarray
        A time series

    Returns
    -------
    int
        First index containing NaN, or length of the time series is it contains no NaN
    
    Examples
    --------
    >>> first_non_finite_index(numpy.array([1, 2, 4, 3, numpy.nan, numpy.nan]).reshape((-1, 1)))
    4
    >>> first_non_finite_index(numpy.array([1, 2, 4, 3, 1.]).reshape((-1, 1)))
    5
    """
    timestamps_infinite = numpy.all(~numpy.isfinite(X), axis=1)  
    # Are there NaNs padded after the TS?
    if numpy.alltrue(~timestamps_infinite):
        idx = X.shape[0]
    else:  # Yes? then return the first index of these NaNs
        idx = numpy.nonzero(timestamps_infinite)[0][0]
    return idx
