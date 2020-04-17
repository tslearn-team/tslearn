from sklearn.neural_network import MLPClassifier, MLPRegressor

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesMLPClassifier(MLPClassifier):
    def fit(self, X, y):
        n_ts = X.shape[0]
        X_ = X.reshape((n_ts, -1))
        return super(TimeSeriesMLPClassifier, self).fit(X_, y)

    def predict(self, X):
        n_ts = X.shape[0]
        X_ = X.reshape((n_ts, -1))
        return super(TimeSeriesMLPClassifier, self).predict(X_)

    def predict_log_proba(self, X):
        n_ts = X.shape[0]
        X_ = X.reshape((n_ts, -1))
        return super(TimeSeriesMLPClassifier, self).predict_log_proba(X_)

    def predict_proba(self, X):
        n_ts = X.shape[0]
        X_ = X.reshape((n_ts, -1))
        return super(TimeSeriesMLPClassifier, self).predict_proba(X_)


class TimeSeriesMLPRegressor(MLPRegressor):
    def fit(self, X, y):
        n_ts = X.shape[0]
        X_ = X.reshape((n_ts, -1))
        return super(TimeSeriesMLPRegressor, self).fit(X_, y)

    def predict(self, X):
        n_ts = X.shape[0]
        X_ = X.reshape((n_ts, -1))
        return super(TimeSeriesMLPRegressor, self).predict(X_)
