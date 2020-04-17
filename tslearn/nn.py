from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils import check_array

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesMLPClassifier(MLPClassifier):
    def fit(self, X, y):
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPClassifier, self).fit(X_, y)

    def predict(self, X):
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPClassifier, self).predict(X_)

    def predict_log_proba(self, X):
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPClassifier, self).predict_log_proba(X_)

    def predict_proba(self, X):
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPClassifier, self).predict_proba(X_)


class TimeSeriesMLPRegressor(MLPRegressor):
    def fit(self, X, y):
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPRegressor, self).fit(X_, y)

    def predict(self, X):
        X_ = check_array(X, force_all_finite=True, allow_nd=True)
        X_ = X_.reshape((X_.shape[0], -1))
        return super(TimeSeriesMLPRegressor, self).predict(X_)
