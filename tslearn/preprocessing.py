from sklearn.base import TransformerMixin

from tslearn.utils import npy3d_time_series_dataset

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesScalerMinMax(TransformerMixin):
    """Scaler for time series. Scales time series so that their span in each dimension is between ``min`` and ``max``.
    
    Example
    -------
    >>> TimeSeriesScalerMinMax(min=1., max=2.).fit_transform([[0, 3, 6]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[ 1. ],
            [ 1.5],
            [ 2. ]]])
    """
    def __init__(self, min=0., max=1.):
        self.min_ = min
        self.max_ = max

    def fit_transform(self, X):
        """Fit to data, then transform it.

        Parameters
        ----------
        X
            Time series dataset to be rescaled

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        X_ = npy3d_time_series_dataset(X)
        for i in range(X_.shape[0]):
            for d in range(X_.shape[2]):
                cur_min = X_[i, :, d].min()
                cur_max = X_[i, :, d].max()
                cur_range = cur_max - cur_min
                X_[i, :, d] = (X_[i, :, d] - cur_min) * (self.max_ - self.min_) / cur_range + self.min_
        return X_


class TimeSeriesScalerMeanVariance(TransformerMixin):
    """Scaler for time series. Scales time series so that their mean (resp. standard deviation) in each dimension is
    ``mu`` (resp. ``std``).
    
    Example
    -------
    >>> TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform([[0, 3, 6]]) # doctest: +NORMALIZE_WHITESPACE
    array([[[-1.22474487],
            [ 0. ],
            [ 1.22474487]]])
    """
    def __init__(self, mu=0., std=1.):
        self.mu_ = mu
        self.std_ = std

    def fit_transform(self, X):
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X
            Time series dataset to be rescaled

        Returns
        -------
        numpy.ndarray
            Rescaled time series dataset
        """
        X_ = npy3d_time_series_dataset(X)
        for i in range(X_.shape[0]):
            for d in range(X_.shape[2]):
                cur_mean = X_[i, :, d].mean()
                cur_std = X_[i, :, d].std()
                X_[i, :, d] = (X_[i, :, d] - cur_mean) * self.std_ / cur_std + self.mu_
        return X_


if __name__ == "__main__":
    import doctest
    doctest.testmod()
