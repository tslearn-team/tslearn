from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
from scipy.sparse import coo_matrix


class NonMyopicEarlyClassification(BaseEstimator, ClassifierMixin):
    """Early Classification modelling for time series using the model presented in [1]

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.

    base_classifier : Estimator
        Estimator (instance) to be cloned and used for classifications.

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

    def __init__(
        self,
        n_clusters,
        base_classifier,
        minimum_time_stamp,
        lamb,
        cost_time_parameter,
    ):
        super(NonMyopicEarlyClassification, self).__init__()
        self.base_classifier = base_classifier
        self.n_clusters = n_clusters
        self.minimum_time_stamp = minimum_time_stamp
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
        y_classes = np.unique(y)
        self.labels_ = sorted(set(y_classes))
        y_classes_indices = [self.labels_.index(yi) for yi in y_classes]
        y_ = y
        for idx, current_classe in enumerate(y_classes):
            y_[y_ == current_classe] = y_classes_indices[idx]

        self.cluster_ = TimeSeriesKMeans(n_clusters=self.n_clusters)
        self.classifiers_ = {
            t: clone(self.base_classifier)
            for t in range(self.minimum_time_stamp, X.shape[1] + 1)
        }
        self.__n_classes_ = len(y_classes_indices)
        self.__len_X_ = X.shape[1]
        self.pyhatyck_ = np.empty((self.__len_X_ - self.minimum_time_stamp + 1,
                                   self.n_clusters, self.__n_classes_, self.__n_classes_))
        c_k = self.cluster_.fit_predict((X))
        X_and_cluster = np.concatenate((X, c_k[:, np.newaxis]), axis=1)
        X1, X2, c_k1, c_k2, y1, y2 = train_test_split(X_and_cluster, c_k, y_, test_size=0.5)
        vector_of_ones = np.ones((len(X[:]),))
        self.pyck_ = coo_matrix(
            (vector_of_ones, (y_, c_k)),
            shape=(self.__n_classes_, self.n_clusters),
        ).toarray()
        self.pyck_ /= self.pyck_.sum(axis=0, keepdims=True)

        for t in range(self.minimum_time_stamp, self.__len_X_ + 1):
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
                        y2_current_cluster, y2_hat, labels=y_classes_indices, normalize="true"
                    )
                    #The diagonal of the confusion matrices must be put to 0
                    # to fullfil the criterions of the cost function
                    np.fill_diagonal(conf_matrix, 0)
                    self.pyhatyck_[t - self.minimum_time_stamp, k] = conf_matrix

    def get_cluster_probas(self, X):
        """
        This function computes the probabilities of the time series xt to be in a cluster of the model

        Parameters
        ----------
        X: vector
            a time series that may be truncated

        Returns
        -------
        vector : the probabilities of the given time series to be in the clusters of the model
        """
        diffs = X[np.newaxis, :] - self.cluster_.cluster_centers_[:, :len(X)]
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

    def _expected_cost(self, X, tau):
        """
        From a incomplete series xt, this function compute the expected cost of a prediction made at time "last time of
        xt + tau"

        Parameters
        ----------
        X: vector
            a time series that may be truncated for which we want to have the cost
        tau: int
            gives the future time for which the cost is calculated

        Returns
        --------
        float : the computed cost
        """
        proba_clusters = self.get_cluster_probas(X=X)
        truncated_t = X.shape[-1]
        sum_pyhatyck = np.sum(self.pyhatyck_[truncated_t + tau - self.minimum_time_stamp], axis=1)
        sum_pyhatyck = np.transpose(sum_pyhatyck)
        sum_global = np.sum((sum_pyhatyck * self.pyck_), axis=0)
        cost = np.dot(proba_clusters, sum_global)
        return cost + self._cost_time(truncated_t + tau)

    def cost_function_minimizer(self, end_of_time, xt):
        """
        We want to minimize a function "funct" according a time series "xt" for a list of integers from 0 to
        "end_of_time"

        Parameters
        ----------
        end_of_time: int
            the highest integer at which the function to be minimized is calculated
        function_to_minimize: funct
            the function of interest
        xt: vector
            the time series related to the function "funct"

        Returns
         ------
         int : The integer in {0,...,stop} that minimizes "funct"
         float : the so fat minimum cost
        """
        costs = [self._expected_cost(xt, tau) for tau in range(end_of_time + 1)]
        tau_star = np.argmin(costs)
        min_cost = costs[tau_star]
        return tau_star, min_cost

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
        length_Xi = Xi.shape[0]
        Xi = np.reshape(Xi, (1, length_Xi))
        time_prediction = self.minimum_time_stamp
        for t in range(self.minimum_time_stamp, self.__len_X_ + 1):
            minimum_tau, minimum_loss = self.cost_function_minimizer(
                end_of_time=length_Xi - t, xt=Xi[:, :t])
            if (time_prediction == length_Xi) or (minimum_tau == t):
                result = self.classifiers_[t].predict(Xi[:, :t])
                result_proba = self.classifiers_[t].predict_proba(Xi[:, :t])
                loss_exit = minimum_loss
                break
            else:
                time_prediction = time_prediction + 1
        return result, time_prediction, result_proba, loss_exit

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

        y_pred = []
        time_prediction = []
        for i in range(0, X.shape[0]):
            (
                new_classif,
                new_time,
                new_proba,
                new_cost,
            ) = self._predict_single_series(X[i])
            y_pred.append(new_classif)
            time_prediction.append(new_time)
        return y_pred, time_prediction

    def _cost_time(self, t):
        return t * self.cost_time_parameter
