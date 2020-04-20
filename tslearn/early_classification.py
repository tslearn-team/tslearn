from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import coo_matrix

from tslearn.utils import to_time_series_dataset, check_dims
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.clustering import TimeSeriesKMeans


class NonMyopicEarlyClassification(BaseEstimator, ClassifierMixin):
    """Early Classification modelling for time series using the model presented in [1]

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.

    base_classifier : Estimator or None
        Estimator (instance) to be cloned and used for classifications.
        If None, the chosen classifier is a 1NN with Euclidean metric.

    minimum_time_stamp : int
        Earliest time at which a classification can be performed on a time series

    lamb : float
        Value of the hyper parameter lambda used during the computation of the cost function to evaluate the probability
        that a time series belongs to a cluster given the time series.

    cost_time_parameter : float
        Parameter of the cost function of time. This function is of the form : f(time) = time * cost_time_parameter

    Attributes
    --------------------
    silhouette_ : float
        The silhouette score from the clustering

    classifiers_ : list
        A list containing all the classifiers trained for the model

    clusters_ : dictionary
        Contains the times series by clusters

    pyhatyck_ : dictionary
        Contains the probabilities of being classified as class y_hat given class y and cluster ck

    indice_ck_ : list
        Contains for each clusters the indexes of the time series in the dataset

    Examples
    --------------------

    References
    --------------------
    [1] A. Dachraoui, A. Bondu & A. Cornuejols. Early classification of time series as a non myopic sequential decision
    making problem. 2015 Conference paper


    """

    def __init__(self, n_clusters=2, base_classifier=None,
                 minimum_time_stamp=0, lamb=1., cost_time_parameter=1.):
        super(NonMyopicEarlyClassification, self).__init__()
        self.base_classifier = base_classifier
        self.n_clusters = n_clusters
        self.min_t = minimum_time_stamp
        self.lamb = lamb
        self.cost_time_parameter = cost_time_parameter

    def fit(self, X, y):
        """
        This function fits classifiers that are currently multilayer perceptrons to a training set of time series and
        associated classes. A classifier is fit for each time stamp above a minimum time stamp that is an attribute of
        the class of the model. Then some probabilities are computed using clusters already computed.
        This function should be divided among 2 or 3 functions.

        Parameters
        ----------
        X: Array-like
            a dataset of time series
        y: vector
            the associated classes of the series from X_train
        """

        X = check_dims(X)
        y_classes = np.unique(y)
        self.labels_ = sorted(set(y_classes))
        y_classes_indices = [self.labels_.index(yi) for yi in y_classes]
        y_ = np.copy(y)
        for idx, current_classe in enumerate(y_classes):
            y_[y_ == current_classe] = y_classes_indices[idx]

        self.cluster_ = TimeSeriesKMeans(n_clusters=self.n_clusters)
        if self.base_classifier is not None:
            clf = self.base_classifier
        else:
            clf = KNeighborsTimeSeriesClassifier(n_neighbors=1,
                                                 metric="euclidean")
        self.classifiers_ = {t: clone(clf)
                             for t in range(self.min_t, X.shape[1] + 1)}
        self.__n_classes_ = len(y_classes_indices)
        self.__len_X_ = X.shape[1]
        self.pyhatyck_ = np.empty((self.__len_X_ - self.min_t,
                                   self.n_clusters,
                                   self.__n_classes_, self.__n_classes_))
        c_k = self.cluster_.fit_predict(X)
        X1, X2, c_k1, c_k2, y1, y2 = train_test_split(X, c_k, y_, test_size=0.5)
        vector_of_ones = np.ones((X.shape[0], ))
        self.pyck_ = coo_matrix(
            (vector_of_ones, (y_, c_k)),
            shape=(self.__n_classes_, self.n_clusters),
        ).toarray()
        self.pyck_ /= self.pyck_.sum(axis=0, keepdims=True)

        for t in range(self.min_t, self.__len_X_):
            self.classifiers_[t].fit(X1[:, :t], y1)
            for k in range(0, self.n_clusters):
                index = (c_k2 == k)
                if index.shape[0] != 0:
                    X2_current_cluster = X2[index, :t]
                    y2_current_cluster = y2[index]
                    y2_hat = self.classifiers_[t].predict(
                        X2_current_cluster[:, :t]
                    )
                    conf_matrix = confusion_matrix(
                        y2_current_cluster, y2_hat, labels=y_classes_indices, normalize="pred"
                    )
                    # pyhatyck_ stores
                    # P_{t+\tau}(\hat{y} | y, c_k) \delta_{y \neq \hat{y}}
                    # elements so it should have a null diagonal because of
                    # the \delta_{y \neq \hat{y}} term
                    np.fill_diagonal(conf_matrix, 0)
                    self.pyhatyck_[t - self.min_t, k] = conf_matrix
        return self

    def get_cluster_probas(self, Xi):
        """
        This function computes the probabilities of the time series xt to be in a cluster of the model

        Parameters
        ----------
        X: vector
            a time series that may be truncated

        Returns
        -------
        vector : the probabilities of the given time series to be in the clusters of the model

        Examples
        --------
        >>> dataset = to_time_series_dataset([[1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [3, 2, 1, 1, 2, 3]])
        >>> y = [0, 1, 0]
        >>> ts0 = to_time_series([1, 2])
        >>> model = NonMyopicEarlyClassification(n_clusters=3, lamb=0.)
        >>> probas = model.fit(dataset, y).get_cluster_probas(ts0)
        >>> probas.shape
        (3, )
        >>> probas
        [.33, .33, .33]
        >>> model = NonMyopicEarlyClassification(n_clusters=3, lamb=10000.)
        >>> probas = model.fit(dataset, y).get_cluster_probas(ts0)
        >>> probas.shape
        (3, )
        >>> probas
        [.5, .5, 0.]
        """
        diffs = Xi[np.newaxis, :] - self.cluster_.cluster_centers_[:, :len(Xi)]
        distances_clusters = np.linalg.norm(diffs, axis=(1, 2))
        minimum_distance = np.min(distances_clusters)
        average_distance = np.mean(distances_clusters)
        minimum_distance_to_average = (
            average_distance - minimum_distance
        ) / average_distance
        delta = (average_distance - distances_clusters) / average_distance
        probas = 1.0 / (
            np.exp(self.lamb * minimum_distance_to_average)
            + np.exp(self.lamb * (minimum_distance_to_average - delta))
            + 1e-6
        )
        return probas / probas.sum()

    def _expected_costs(self, Xi):
        """Compute expected future costs from an incoming time series `Xi`.

        This cost is computed, for a time horizon :math:`\tau`, as:

        ..math ::

            \sum_k P(c_k | Xi) \sum_y P(y | c_k)
                \sum_\hat{y}
                P_{t+\tau}(\hat{y} | y, c_k) \delta_{y \neq \hat{y}}

        where:

        * :math:`P(c_k | Xi)` is obtained through a call to
        `get_cluster_probas`
        * :math:`P(y | c_k)` is stored in `pyck_`
        * :math:`P_{t+\tau}(\hat{y} | y, c_k) \delta_{y \neq \hat{y}}` is
        stored in `pyhatyck_`

        Parameters
        ----------
        Xi: numpy array, shape (t, d)
            A time series observed up to time t

        Returns
        --------
        cost : numpy array of shape (self.__len_X_ - t + 1, )
            Expected future costs for all time stamps from t to self.__len_X_
        """
        proba_clusters = self.get_cluster_probas(Xi=Xi)
        truncated_t = Xi.shape[0]
        # pyhatyck_ is indexed by: t, k, y, yhat
        sum_pyhatyck = np.sum(
            self.pyhatyck_[truncated_t - self.min_t:],
            axis=-1
        )
        sum_pyhatyck = np.transpose(sum_pyhatyck, axes=(0, 2, 1))
        # sum_pyhatyck is now indexed by: t, y, k
        sum_global = np.sum(sum_pyhatyck * self.pyck_[np.newaxis, :], axis=1)
        cost = np.dot(sum_global, proba_clusters)
        return cost + self._cost_time(np.arange(truncated_t, self.__len_X_ + 1))

    def _predict_single_series(self, Xi):
        """
        This function classifies a single time series xt

        Parameters
        ----------
        xt: vector
            a time series that is probably incomplete but that nonetheless we want to classify
        Returns
        -------
        int: the class which is predicted
        int : the time of the prediction
        float : the probability used for computing the cost
        float : the loss when classifying
        """

        time_prediction = self.min_t
        for t in range(self.min_t, self.__len_X_ + 1):
            tau_star = np.argmin(self._expected_costs(Xi=Xi[:t]))
            if (t == self.__len_X_) or (tau_star == t):
                result = self.classifiers_[t].predict([Xi[:t]])
                result_proba = self.classifiers_[t].predict_proba([Xi[:t]])
                break
        return result, time_prediction, result_proba

    def predict(self, X):
        """
        Predicts the classes of the series of the test dataset.

        Parameters
        ----------
        X_test : Array-like
            The test dataset

        Returns
        -------
        Vector : the predicted classes
        """

        X = check_dims(X)
        y_pred = []
        time_prediction = []
        for i in range(0, X.shape[0]):
            cl, t, proba = self._predict_single_series(X[i])
            y_pred.append(cl)
            time_prediction.append(t)
        return y_pred, time_prediction

    def _cost_time(self, t):
        return t * self.cost_time_parameter
