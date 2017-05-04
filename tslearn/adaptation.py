import numpy
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d

from tslearn.metrics import dtw_path

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class DTWSampler(BaseEstimator, TransformerMixin):
    """A class for non-linear DTW-based resampling of time series.
    The principle is to use a modality (or a set of modalities) to perform DTW alignment with respect to a reference
    and then resample other modalities using the obtained DTW path.

    A typical usage should be:
    1- build the sampler by calling the constructor
    2- fit (i.e. provide reference time series)
    3- call prepare_transform to perform DTW between base modalities of the targets and those of the reference.
    4- call transform to get resampled time series for all other modalities"""
    def __init__(self, n_samples=100, interp_kind="slinear"):
        self.n_samples = n_samples
        self.interp_kind = interp_kind
        self.reference_series_ = None

        self.saved_dtw_paths_ = None

    def prepare_transform(self, ts_to_be_rescaled):
        """Prepare the model for temporal resampling by computing DTW alignment path between the reference time series
        and a time series to be rescaled or a set of time series to be rescaled.
        If ts_to_be_rescaled contains a single time series, all series from the dataset will be rescaled using the
        DTW path between that time series and the reference one, otherwise, the X array given at transform time should
        have the same number of time series (X.shape[0]) as ts_to_be_rescaled."""
        if ts_to_be_rescaled.ndim == 1:
            ts_to_be_rescaled = ts_to_be_rescaled.reshape((1, -1, 1))
        elif ts_to_be_rescaled.ndim == 2:
            if ts_to_be_rescaled.shape[1] == self.reference_series_.shape[1]:
                ts_to_be_rescaled = ts_to_be_rescaled.reshape((1, -1, ts_to_be_rescaled.shape[1]))
            else:
                ts_to_be_rescaled = ts_to_be_rescaled.reshape((ts_to_be_rescaled.shape[0], -1, 1))
        elif ts_to_be_rescaled.ndim >= 4:
            raise ValueError
        # Now ts_to_be_rescaled is of shape n_ts, sz, d with d = self.reference_series.shape[-1]
        self.saved_dtw_paths_ = []
        for ts in ts_to_be_rescaled:
            end = last_index(ts)
            ts_resampled = resampled(ts[:end], n_samples=self.n_samples, kind=self.interp_kind)
            path, d = dtw_path(self.reference_series_, ts_resampled)
            self.saved_dtw_paths_.append(path)

    def fit(self, X):
        """Register X as the reference series and interpolate it to get a series of size self.nsamples.
        X should contain a single (possibly multivariate) time series."""
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        elif X.ndim == 2:
            if X.shape[0] == 1:
                X = X.reshape((-1, 1))
        elif X.ndim == 3 and X.shape[0] == 1:
            X = X.reshape((-1, 1))
        else:
            raise ValueError

        end = last_index(X)
        self.reference_series_ = resampled(X[:end], n_samples=self.n_samples, kind=self.interp_kind)
        return self

    def transform(self, X):
        assert X.shape[0] == len(self.saved_dtw_paths_)
        X_resampled = numpy.zeros((X.shape[0], self.n_samples, X.shape[2]))
        xnew = numpy.linspace(0, 1, self.n_samples)
        for i in range(X.shape[0]):
            end = last_index(X[i])
            X_resampled[i] = resampled(X[i, :end], n_samples=self.n_samples, kind=self.interp_kind)
            # Compute indices based on alignment of dimension self.scaling_col_idx with the reference
            indices_xy = [[] for _ in range(self.n_samples)]

            if len(self.saved_dtw_paths_) == 1:
                path = self.saved_dtw_paths_[0]
            else:
                path = self.saved_dtw_paths_[i]

            for t_current, t_ref in path:
                indices_xy[t_ref].append(t_current)
            for j in range(X.shape[2]):
                ynew = numpy.array([numpy.mean(X_resampled[i, indices, j]) for indices in indices_xy])
                X_resampled[i, :, j] = ynew
        return X_resampled


def resampled(X, n_samples=100, kind="linear"):
    if X.ndim == 1:
        X = X.reshape((-1, 1))
    assert X.ndim == 2
    X_out = numpy.zeros((n_samples, X.shape[-1]))
    xnew = numpy.linspace(0, 1, n_samples)
    for di in range(X.shape[-1]):
        f = interp1d(numpy.linspace(0, 1, X.shape[0]), X[:, di], kind=kind)
        X_out[:, di] = f(xnew)
    return X_out


def last_index(X):
    timestamps_infinite = numpy.all(~numpy.isfinite(X), axis=1)  # Are there NaNs padded after the TS?
    if numpy.alltrue(~timestamps_infinite):
        idx = X.shape[0]
    else:  # Yes? then remove them
        idx = numpy.nonzero(timestamps_infinite)[0][0]
    return idx
