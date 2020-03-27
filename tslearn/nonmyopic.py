from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import roc_auc_score
import os.path


class NonMyopicEarlyClassification(BaseEstimator, ClassifierMixin):
    """Early Classification modelling for time series using the model presented in [1]

    Parameters
    ----------
    number_cluster : int
        Number of clusters to form.

    solver :
        The method of gradient descent used during training

    hidden_layer_sizes : tuple of int
        Size of the hidden layers

    random_state : int
        Seed of the MLP

    maximum_iteration : int
        Maximum number of iterations performed by the MLP

    minimum_time_stamp : int
        Earliest time at which a classification can be performed on a time series

    lamb : float
        Value of the hyper parameter lambda

    cost_time_parameter : float
        Parameter of the cost function of time

    Attributes
    --------------------
    silhouette_ : float
        The silhouette score from the clustering

    classifier_ : list
        A list containing all the classifiers trained for the model

    clusters_ : dictionary
        Contains the times series by clusters

    pyhatyck_ : dictionary
        Contains the probabilities of being classified as class y_hat given class y and cluster ck

    indice_ck_ : list
        Contains for each clusters the indexes of the time series in the dataset

    number_classes_ : int
        The number of classes in the series

    len_X_ : int
        The length of the time series

    Examples
    --------------------
    >>>

    References
    --------------------
    [1] A. Dachraoui, A. Bondu & A. Cornuéjols. Early classification of time series as a non myopic sequential decision
    making problem. 2015 Conference paper


    """

    def __init__(
        self,
        number_cluster,
        solver,
        hidden_layer_sizes,
        random_state,
        maximum_iteration,
        minimum_time_stamp,
        lamb,
        cost_time_parameter,
    ):
        super(NonMyopicEarlyClassification, self).__init__()
        self.solver = solver
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.maximum_iteration = maximum_iteration
        self.number_cluster = number_cluster
        self.minimum_time_stamp = minimum_time_stamp
        self.lamb = lamb
        self.cost_time_parameter = cost_time_parameter

    def clustering(self, X):
        """
        This function compute and fit clusters for a training dataset of time series using K-means and euclidean
        distances
        Parameters
        ----------
        X: array-like
            The training dataset of time series

        Returns
        -------
        Vector : a vector as long as the number of series in the training dataset that indicates in which cluster the
        series at the same index belongs

         Float : the silhouette score of the fitting of the clustering
        """
        self.cluster_ = TimeSeriesKMeans(n_clusters=self.number_cluster)
        c_k = self.cluster_.fit_predict((to_time_series_dataset(X)))
        silhouette = silhouette_score(
            to_time_series_dataset(X), c_k, metric="euclidean"
        )

        return c_k, silhouette

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
        c_k, silhouette = self.clustering(X)
        self.silhouette_ = silhouette
        self.classifier_ = []
        self.clusters_ = {}
        self.pyhatyck_ = {}
        self.indice_ck_ = []
        self.number_classes_ = len(classes_y)
        self.len_X_ = X.shape[1]
        mid_X = int(X.shape[0] / 2)
        X1 = X[:mid_X]
        Y1 = Y[:mid_X]
        X2 = X[mid_X:]
        Y2 = Y[mid_X:]
        c_k2 = c_k[mid_X:]
        self.pyck_ = np.zeros(shape=(self.number_classes_, self.number_cluster))
        for current_cluster in range(0, self.number_cluster):
            self.clusters_["ck_cm{0}".format(current_cluster)] = []
            self.pyhatyck_["pyhatycks{0}".format(current_cluster)] = []
            self.indice_ck_.append(np.where(c_k == current_cluster))
            Y_current_cluster = Y[self.indice_ck_[-1]]
            for current_class in range(0, self.number_classes_):
                value_current_class = classes_y[current_class]
                if len(Y_current_cluster) == 0:
                    current_pyck = 0
                else:
                    current_pyck = len(
                        Y_current_cluster[Y_current_cluster == value_current_class]
                    ) / len(Y_current_cluster)
                self.pyck_[current_class][current_cluster] = current_pyck

        for i in range(self.minimum_time_stamp, X.shape[1] + 1):
            self.classifier_.append(
                MLPClassifier(
                    solver=self.solver,
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    random_state=self.random_state,
                    max_iter=self.maximum_iteration,
                )
            )
            self.classifier_[-1].fit(X1[:, :i], Y1)
            for current_cluster in range(0, self.number_cluster):
                index_cluster = np.where(c_k2 == current_cluster)
                if len(index_cluster[0]) != 0:
                    X2_current_cluster = np.squeeze(X2[index_cluster, :i], axis=0)
                    Y2_current_cluster = Y2[tuple(index_cluster)]
                    Y2_hat = self.classifier_[-1].predict(X2_current_cluster[:, :i])
                    self.clusters_["ck_cm{0}".format(current_cluster)].append(
                        confusion_matrix(Y2_current_cluster, Y2_hat, labels=classes_y)
                    )
                    column_sum = self.clusters_["ck_cm{0}".format(current_cluster)][
                        -1
                    ].sum(1)
                    for current_class in range(0, self.number_classes_):
                        if column_sum[current_class] == 0:
                            column_sum[current_class] = 1
                    matrix_proba_current_cluster_class = np.asarray(
                        [
                            [
                                self.clusters_["ck_cm{0}".format(current_cluster)][-1][
                                    0
                                ][0]
                                / column_sum[0],
                                self.clusters_["ck_cm{0}".format(current_cluster)][-1][
                                    0
                                ][1]
                                / column_sum[0],
                            ],
                            [
                                self.clusters_["ck_cm{0}".format(current_cluster)][-1][
                                    1
                                ][0]
                                / column_sum[1],
                                self.clusters_["ck_cm{0}".format(current_cluster)][-1][
                                    1
                                ][1]
                                / column_sum[1],
                            ],
                        ]
                    )
                    self.pyhatyck_["pyhatycks{0}".format(current_cluster)].append(
                        matrix_proba_current_cluster_class
                    )

    def get_avg_dist(self, X):
        """
        Compute the average euclidean distance of a given time series to the centroid of each cluster of the model

        Parameters
        ----------
        X: Vector
            A time series that may  be truncated

        Returns
        -------
        Float : the average euclidean distance of the time series to the centroid of each cluster of the model
        Float : the minimum distance between the centroid of a cluster and the time series
        """

        sum_distances = 0
        minimum_distance = 0
        for current_cluster in range(0, self.number_cluster):
            euclidean_distance = np.linalg.norm(
                X - self.cluster_.cluster_centers_[current_cluster, : len(X)]
            )
            sum_distances = sum_distances + euclidean_distance
            if minimum_distance < euclidean_distance:
                minimum_distance = euclidean_distance
        return sum_distances / self.number_cluster, minimum_distance

    def get_dist(self, X, wanted_cluster):
        """
        Compute the euclidean distance between a time series xt and the centroid of a cluster of the model
        Parameters
        ----------
        X: vector
            a time series that may be truncated
        wanted_cluster: int
            the number of the cluster for which the distance is wanted

        Returns
        -------
        Float: the expected euclidean distance
        """

        euclidean_dist = np.linalg.norm(
            X - self.cluster_.cluster_centers_[wanted_cluster, : len(X)]
        )

        return euclidean_dist

    def get_cluster_prob(self, X):
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
        average_distance, minimum_distance = self.get_avg_dist(X)
        minimum_distance_to_average = (
            average_distance - minimum_distance
        ) / average_distance
        sum_sk = 0
        sk = []
        for current_cluster in range(0, self.number_cluster):
            d_k = self.get_dist(X, current_cluster)
            Delta_k = (average_distance - d_k) / average_distance
            proba_current_cluster = 1 / (
                math.exp(self.lamb * minimum_distance_to_average)
                + math.exp(self.lamb * (minimum_distance_to_average - Delta_k))
                + 10e-6
            )
            sk.append(proba_current_cluster)
            sum_sk = sum_sk + sk[-1]
        final_sk = [(x / sum_sk) for x in sk]
        return final_sk

    def cost_function(self, X, tau):
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
        -------
        float : the computed cost
        """
        cost = 0
        proba_clusters = self.get_cluster_prob(X=X)
        truncated_t = X.shape[-1]
        for current_cluster in range(0, self.number_cluster):
            cost_y = 0
            for y in range(0, self.number_classes_):
                cost_y_hat = 0
                for y_hat in range(0, self.number_classes_):
                    if y != y_hat:
                        cost_y_hat = (
                            cost_y_hat
                            + self.pyhatyck_["pyhatycks{0}".format(current_cluster)][
                                truncated_t + tau - self.minimum_time_stamp
                            ][y][y_hat]
                        )
                cost_y = cost_y + self.pyck_[y, current_cluster] * cost_y_hat
            cost = cost + proba_clusters[current_cluster] * cost_y
        cost = cost + self.cost_time(t=truncated_t + tau)

        return cost

    @staticmethod
    def minimize_integer(end_of_time, function_to_minimize, xt):
        """
        We want to minimize a function "funct" according a time series "xt" for a list of integers from 0 to "stop"
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

    def classification(self, xt, cost_function):
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
                result = self.classifier_[-1].predict(xt)
                result_proba = self.classifier_[-1].predict_proba(xt)
                stop_criterion = True
                minimum_tau, minimum_loss = self.minimize_integer(
                    end_of_time=time_stamp - time_prediction,
                    function_to_minimize=cost_function,
                    xt=xt[:, :time_prediction],
                )
                loss_exit = minimum_loss
            elif minimum_tau == time_prediction:
                result = self.classifier_[
                    time_prediction - self.minimum_time_stamp
                ].predict(xt[:, :time_prediction])
                result_proba = self.classifier_[
                    time_prediction - self.minimum_time_stamp
                ].predict_proba(xt[:, :time_prediction])
                loss_exit = minimum_loss
                stop_criterion = True
            else:
                time_prediction = time_prediction + 1
        return result, time_prediction, result_proba, loss_exit

    @staticmethod
    def avg_prediction(time_predictions):

        avgerage_time = sum(time_predictions) / len(time_predictions)
        variance_time = np.var(time_predictions)
        return avgerage_time, variance_time

    @staticmethod
    def accuracy_prediction(Y_test, Y_pred_test):

        good_prediction = 0
        for class_index in range(0, len(Y_test)):
            if Y_test[class_index] == Y_pred_test[class_index]:
                good_prediction = good_prediction + 1
        return good_prediction / len(Y_test)

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

        for t in range(0, X.shape[0]):
            new_classif, new_time, new_proba, new_cost = self.classification(
                X[t], self.cost_function
            )
            Y_pred.append(new_classif)
        return Y_pred

    def predict_completed(self, X, Y):
        """
        Predicts the classes of the series of the test dataset and has also access the the true classes to establish
        comparisons

        Parameters
        ----------
        X: Array-like
            The data set that needs to be predict
        Y: Array-vector
            The true classes

        Returns
        -------
        float : average time of prediction
        float : variance of the time of prediction
        float : AUC
        float : average cost
        """
        Y_pred = []
        time_predictions = []
        pred_proba = []
        cost_average = 0

        for current_serie in range(0, X.shape[0]):
            new_classif, new_time, new_proba, new_cost = self.classification(
                X[current_serie], self.cost_function
            )
            Y_pred.append(new_classif)
            time_predictions.append(new_time)
            pred_proba.append(new_proba)
            cost_average = cost_average + new_cost
        average_pred, variance_pred = self.avg_prediction(time_predictions)
        pred_proba = np.reshape(np.asarray(pred_proba), (X.shape[0], 2))
        auc = roc_auc_score(Y, pred_proba[:, 1])
        cost_average = cost_average / X.shape[0]
        return average_pred, variance_pred, auc, cost_average

    def graphic(self, path, name, prediction_time=None):
        plt.clf()
        for current_cluster in range(0, self.number_cluster):
            for current_class in range(0, self.number_classes_):
                for current_hat_class in range(0, self.number_classes_):
                    if current_class != current_hat_class:
                        sum_prob = [
                            self.pyck_[current_class][current_cluster]
                            * item[current_class][current_hat_class]
                            for item in self.pyhatyck_[
                                "pyhatycks{0}".format(current_cluster)
                            ]
                        ]
                        lab = "y={} yhat={} ck={}".format(
                            current_class, current_hat_class, current_cluster
                        )
                        plt.plot(sum_prob, label=lab)
        plt.legend()
        if prediction_time is not None:
            plt.title("The prediction time is " + str(prediction_time))
        plt.savefig(os.path.join(path, name + ".png"), dpi=500)

    def cost_time(self, t):
        time_stamp = np.linspace(0, self.len_X_, num=self.len_X_)
        return time_stamp[t] * self.cost_time_parameter
