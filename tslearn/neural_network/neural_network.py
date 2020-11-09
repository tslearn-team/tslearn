from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils import check_array

from ..bases import TimeSeriesBaseEstimator

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesMLPClassifier(MLPClassifier, TimeSeriesBaseEstimator):
    """A Multi-Layer Perceptron classifier for time series.

    This class mainly reshapes data so that it can be fed to `scikit-learn`'s
    ``MLPClassifier``.

    It accepts the exact same hyper-parameters as ``MLPClassifier``, check
    `scikit-learn docs <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`__
    for a list of parameters and attributes.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=30, sz=16, d=2, n_blobs=3,
    ...                          random_state=0)
    >>> mlp = TimeSeriesMLPClassifier(hidden_layer_sizes=(64, 64),
    ...                               random_state=0)
    >>> mlp.fit(X, y)  # doctest: +ELLIPSIS
    TimeSeriesMLPClassifier(...)
    >>> [c.shape for c in mlp.coefs_]
    [(32, 64), (64, 64), (64, 3)]
    >>> [c.shape for c in mlp.intercepts_]
    [(64,), (64,), (3,)]
    """
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, ) or (n_ts, dim_y)
            Target values.

        Returns
        -------
        TimeSeriesMLPClassifier
            The fitted estimator
        """
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPClassifier, self).fit(X_, y)

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, )
            Array of predicted class labels
        """
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPClassifier, self).predict(X_)

    def predict_log_proba(self, X):
        """Predict the class log-probabilities for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, n_classes)
            Array of predicted class log-probabilities
        """
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPClassifier, self).predict_log_proba(X_)

    def predict_proba(self, X):
        """Predict the class probabilities for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, n_classes)
            Array of predicted class probabilities
        """
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPClassifier, self).predict_proba(X_)


class TimeSeriesMLPRegressor(MLPRegressor, TimeSeriesBaseEstimator):
    """A Multi-Layer Perceptron regressor for time series.

    This class mainly reshapes data so that it can be fed to `scikit-learn`'s
    ``MLPRegressor``.

    It accepts the exact same hyper-parameters as ``MLPRegressor``, check
    `scikit-learn docs <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html>`__
    for a list of parameters and attributes.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> mlp = TimeSeriesMLPRegressor(hidden_layer_sizes=(64, 64),
    ...                               random_state=0)
    >>> mlp.fit(X=[[1, 2, 3], [1, 1.2, 3.2], [3, 2, 1]],
    ...         y=[0, 0, 1])  # doctest: +ELLIPSIS
    TimeSeriesMLPRegressor(...)
    >>> [c.shape for c in mlp.coefs_]
    [(3, 64), (64, 64), (64, 1)]
    >>> [c.shape for c in mlp.intercepts_]
    [(64,), (64,), (1,)]
    """
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Training data.
        y : array-like, shape (n_ts, ) or (n_ts, dim_y)
            Target values.

        Returns
        -------
        TimeSeriesMLPRegressor
            The fitted estimator
        """
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPRegressor, self).fit(X_, y)

    def predict(self, X):
        """Predict the target for the provided data

        Parameters
        ----------
        X : array-like, shape (n_ts, sz, d)
            Test samples.

        Returns
        -------
        array, shape = (n_ts, ) or (n_ts, dim_y)
            Array of predicted targets
        """
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPRegressor, self).predict(X_)
