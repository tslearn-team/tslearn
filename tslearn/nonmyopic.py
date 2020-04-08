from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, clone
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

    __n_classes_ : int private
        The number of classes in the series

    __len_X_ : int
        The length of the time series

    Examples
    --------------------

    References
    --------------------
    [1] A. Dachraoui, A. Bondu & A. Cornuéjols. Early classification of time series as a non myopic sequential decision
    making problem. 2015 Conference paper


    """

    def __init__(
        self,
        n_clusters,
        base_classifier,
        minimum_time_stamp,
        lamb,
        cost_time_parameter
    ):
        super(NonMyopicEarlyClassification, self).__init__()
        self.base_classifier = base_classifier
        self.n_clusters = n_clusters
        self.minimum_time_stamp = minimum_time_stamp
        self.lamb = lamb
        self.cost_time_parameter = cost_time_parameter

    def fit(self, X, Y):
        """
        This function fits classifiers that are currently multilayer perceptrons to a training set of time series and
        associated classes. A classifier is fit for each time stamp above a minimum time stamp that is an attribute of
        the class of the model. Then some probabilities are computed using clusters already computed.
        This function should be divided among 2 or 3 functions.

        Parameters
        ----------
        X: Array-like
            a dataset of time series
        Y: vector
            the associated classes of the series from X_train
        """
        classes_y = np.unique(Y)
        self.cluster_ = TimeSeriesKMeans(n_clusters=self.n_clusters)
        self.classifiers_ = {t: clone(self.base_classifier)
                             for t in range(self.minimum_time_stamp,
                                            X.shape[1] + 1)}
        self.clusters_ = {}
        self.pyhatyck_ = {}
        self.indice_ck_ = []
        self.__n_classes_ = len(classes_y)
        self.__len_X_ = X.shape[1]
        c_k = self.cluster_.fit_predict((to_time_series_dataset(X)))
        mid_X = int(X.shape[0] / 2)
        X1 = X[:mid_X]
        Y1 = Y[:mid_X]
        X2 = X[mid_X:]
        Y2 = Y[mid_X:]
        c_k2 = c_k[mid_X:]
        vector_of_ones = np.ones((len(X[:]),))
        self.pyck_ = coo_matrix((vector_of_ones, (Y, c_k)), shape=(self.__n_classes_, self.n_clusters)).toarray()
        for k in range(0, self.n_clusters):
            self.clusters_["ck_cm{0}".format(k)] = []
            self.pyhatyck_["pyhatycks{0}".format(k)] = []
            self.indice_ck_.append(np.where(c_k == k))
            current_sum = self.pyck_[:, k].sum()
            for current_classe in range(0, self.__n_classes_):
                self.pyck_[current_classe, k] = self.pyck_[current_classe, k] / current_sum

        for t in range(self.minimum_time_stamp, X.shape[1] + 1):
            self.classifiers_[t].fit(X1[:, :t], Y1)
            for k in range(0, self.n_clusters):
                index_cluster = np.where(c_k2 == k)
                if len(index_cluster[0]) != 0:
                    X2_current_cluster = np.squeeze(X2[index_cluster, :t], axis=0)
                    Y2_current_cluster = Y2[tuple(index_cluster)]
                    Y2_hat = self.classifiers_[t].predict(X2_current_cluster[:, :t])
                    self.clusters_["ck_cm{0}".format(k)].append(
                        confusion_matrix(Y2_current_cluster, Y2_hat, labels=classes_y)
                    )
                    column_sum = self.clusters_["ck_cm{0}".format(k)][
                        -1
                    ].sum(1)
                    for current_class in range(0, self.__n_classes_):
                        if column_sum[current_class] == 0:
                            column_sum[current_class] = 1
                    matrix_proba_current_cluster_class = np.asarray(
                        [
                            [
                                self.clusters_["ck_cm{0}".format(k)][-1][
                                    0
                                ][0]
                                / column_sum[0],
                                self.clusters_["ck_cm{0}".format(k)][-1][
                                    0
                                ][1]
                                / column_sum[0],
                            ],
                            [
                                self.clusters_["ck_cm{0}".format(k)][-1][
                                    1
                                ][0]
                                / column_sum[1],
                                self.clusters_["ck_cm{0}".format(k)][-1][
                                    1
                                ][1]
                                / column_sum[1],
                            ],
                        ]
                    )
                    self.pyhatyck_["pyhatycks{0}".format(k)].append(
                        matrix_proba_current_cluster_class
                    )

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
        distances_clusters = []
        centroids = self.cluster_.cluster_centers_[:, :len(X)]
        for k in range(0, self.n_clusters):
            euclidean_dist = np.linalg.norm(
                X - centroids[k]
            )
            distances_clusters.append(euclidean_dist)
        distances_clusters = np.asarray(distances_clusters)
        minimum_distance = np.min(distances_clusters)
        average_distance = np.mean(distances_clusters)
        minimum_distance_to_average = (
                                              average_distance - minimum_distance
                                      ) / average_distance
        sum_sk = 0
        sk = []
        for k in range(0, self.n_clusters):
            d_k = distances_clusters[k]
            Delta_k = (average_distance - d_k) / average_distance
            proba_current_cluster = 1 / (
                np.exp(self.lamb * minimum_distance_to_average)
                + np.exp(self.lamb * (minimum_distance_to_average - Delta_k))
                + 10e-6
            )
            sk.append(proba_current_cluster)
            sum_sk = sum_sk + sk[-1]
        final_sk = [(x / sum_sk) for x in sk]
        return final_sk

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
        cost = 0
        proba_clusters = self.get_cluster_probas(X=X)
        truncated_t = X.shape[-1]
        for k in range(0, self.n_clusters):
            cost_y = 0
            for y in range(0, self.__n_classes_):
                cost_y_hat = 0
                for y_hat in range(0, self.__n_classes_):
                    if y != y_hat:
                        cost_y_hat = (
                            cost_y_hat
                            + self.pyhatyck_["pyhatycks{0}".format(k)][
                                truncated_t + tau - self.minimum_time_stamp
                            ][y][y_hat]
                        )
                cost_y = cost_y + self.pyck_[y, k] * cost_y_hat
            cost = cost + proba_clusters[k] * cost_y
        cost = cost + self.cost_time(t=truncated_t + tau)

        return cost

    @staticmethod
    def minimize_integer(end_of_time, function_to_minimize, xt):
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
        tau_star = 0
        so_far_minimum_cost = function_to_minimize(xt, tau_star)
        for tau in range(1, end_of_time):
            current_cost = function_to_minimize(xt, tau)
            if current_cost < so_far_minimum_cost:
                so_far_minimum_cost = current_cost
                tau_star = tau
        return tau_star + xt.shape[-1], so_far_minimum_cost

    def _predict_at_fixed_length(self, xt, cost_function):
        """
        This function classifies a single time series xt

        Parameters
        ----------
        xt: vector
            a time series that is probably incomplete but that nonetheless we want to classify
        cost_function: funct
            The cost function that will be use to achieve our classification purpose
        Returns
        -------
        int: the class which is predicted
        int : the time of the prediction
        float : the probability used for computing the cost
        float : the loss when classifying
        """
        time_stamp = xt.shape[0]
        xt = np.reshape(xt, (1, time_stamp))
        time_prediction = self.minimum_time_stamp
        stop_criterion = False
        while stop_criterion is False:
            minimum_tau, minimum_loss = self.minimize_integer(
                end_of_time=time_stamp - time_prediction,
                function_to_minimize=cost_function,
                xt=xt[:, :time_prediction],
            )
            if time_prediction == time_stamp:
                result = self.classifiers_[time_prediction].predict(xt)
                result_proba = self.classifiers_[time_prediction].predict_proba(xt)
                stop_criterion = True
                minimum_tau, minimum_loss = self.minimize_integer(
                    end_of_time=time_stamp - time_prediction,
                    function_to_minimize=cost_function,
                    xt=xt[:, :time_prediction],
                )
                loss_exit = minimum_loss
            elif minimum_tau == time_prediction:
                result = self.classifiers_[
                    time_prediction - self.minimum_time_stamp
                ].predict(xt[:, :time_prediction])
                result_proba = self.classifiers_[
                    time_prediction - self.minimum_time_stamp
                ].predict_proba(xt[:, :time_prediction])
                loss_exit = minimum_loss
                stop_criterion = True
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

        Y_pred = []
        time_prediction = []
        for i in range(0, X.shape[0]):
            new_classif, new_time, new_proba, new_cost = self._predict_at_fixed_length(
                X[i], self.cost_function
            )
            Y_pred.append(new_classif)
            time_prediction.append(new_time)
        return Y_pred, time_prediction

    def _cost_time(self, t):
        return t * self.cost_time_parameter
