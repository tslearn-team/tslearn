from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_data
import math
from sklearn.metrics import roc_auc_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cost_time_parameter', type=float, default=0.01)
args = parser.parse_args()


class NonMyopic(BaseEstimator, ClassifierMixin):
    """
    >>> X_train, Y_train = generate_data(S=2500, t=50, omeg2=10.3, omeg1=10, sd=0.5)
    >>> model = NonMyopic(number_cluster=2, solver="sgd", hidden_layer_sizes=26, random_state=1, maximum_iteration=400, minimum_time_stamp=30, lamb=10)
    >>> model.fit(X_train, Y_train)
    12
    >>> X_test, Y_test = generate_data(S=100, t=50, omeg2=10.3, omeg1=10, sd=3)
    >>> model.classification(X_test[1], model.cost_function)
    (array([1]), 7)
    """
    def __init__(self, number_cluster, solver, hidden_layer_sizes, random_state, maximum_iteration, minimum_time_stamp,
                 lamb, cost_time_parameter):
        super(NonMyopic, self).__init__()
        self.solver = solver
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.maximum_iteration = maximum_iteration
        self.number_cluster = number_cluster
        self.minimum_time_stamp = minimum_time_stamp
        self.lamb = lamb
        self.cost_time_parameter = cost_time_parameter

    def clustering(self, X_train):
        """
        This function compute and fit clusters for a training dataset of time series using K-means and euclidean
        distances
        :param X_train: a training dataset of time series
        :return: a vector as long as the number of series in the training dataset that indicates in which cluster the
        series at the same index belongs, the silhouette score of the fitting of the clustering
        """
        self.cluster_ = TimeSeriesKMeans(n_clusters=self.number_cluster)
        c_k = self.cluster_.fit_predict((to_time_series_dataset(X_train)))
        silhouette = silhouette_score(to_time_series_dataset(X_train), c_k, metric="euclidean")

        return c_k, silhouette

    @staticmethod
    def cross_validation(X_train, Y_train, fold, set_lamb, set_clus, cost_time_parameter):
        index_sta = 0
        index_end = index_sta + int(len(X_train) / fold)
        retain = []
        for fol in range(0, fold):
            X_train_cv = np.concatenate((X_train[:index_sta], X_train[index_end:]), axis=0)
            Y_train_cv = np.concatenate((Y_train[:index_sta], Y_train[index_end:]), axis=0)
            X_test_cv = X_train[index_sta:index_end]
            Y_test_cv = Y_train[index_sta:index_end]
            total_cost = []
            for lamb in set_lamb:
                for clus in set_clus:
                    model = NonMyopic(number_cluster=clus, solver="sgd", hidden_layer_sizes=26, random_state=100,
                                      maximum_iteration=1000,
                                      minimum_time_stamp=4, lamb=lamb, cost_time_parameter=cost_time_parameter)
                    model.fit(X_train_cv, Y_train_cv)
                    avg_pred, var_pred, auc, cost = model.predict(X_test_cv, Y_test_cv)
                    total_cost.append([cost, clus, lamb])
            total_cost = np.asarray(total_cost)
            tot_cost = [item[0] for item in total_cost]
            min_cost_index = np.argmin(tot_cost)
            retain.append(total_cost[min_cost_index])
            index_sta = index_end + 1
            index_end = index_end + int(len(X_train) / fold)
        fin_lamb = [item[-1] for item in retain]
        fin_clus = [item[1] for item in retain]
        final_lamb = sum(fin_lamb) / len(retain)
        final_clus = int(sum(fin_clus) / len(retain))

        return final_clus, final_lamb

    def fit(self, X_train, Y_train):
        """
        This function fits classifiers that are currently multilayer perceptrons to a training set of time series and
        associated classes. A classifier is fit for each time stamp above a minimum time stamp that is an attribute of
        the class of the model. Then some probabilities are computed using clusters already computed.
        This function should be divided among 2 or 3 functions.
        :param X_train: a dataset of time series
        :param Y_train: the associated classes of the series from X_train
        :return: the number 12
        """
        classes_y = np.unique(Y_train)
        c_k, silhouette = self.clustering(X_train)
        self.silhouette_ = silhouette
        self.classifier_ = []
        self.clusters_ = {}
        self.pyhatyck_ = {}
        self.indice_ck_ = []
        self.number_classes_ = len(classes_y)
        self.len_X_ = X_train.shape[1]
        mid_X = int(X_train.shape[0] / 2)
        X_train1 = X_train[:mid_X]
        Y_train1 = Y_train[:mid_X]
        X_train2 = X_train[mid_X:]
        Y_train2 = Y_train[mid_X:]
        c_k2 = c_k[mid_X:]
        self.pyck_ = np.zeros(shape=(self.number_classes_, self.number_cluster))
        for current_cluster in range(0, self.number_cluster):
            #Maybe shall right indice_ck_cm rather than ck_cm, dunno what ck_cm is
            self.clusters_["ck_cm{0}".format(current_cluster)] = []
            self.pyhatyck_["pyhatycks{0}".format(current_cluster)] = []
            self.indice_ck_.append(np.where(c_k == current_cluster))
            Y_train_current_cluster = Y_train[self.indice_ck_[-1]]
            for current_class in range(0, self.number_classes_):
                value_current_class = classes_y[current_class]
                if len(Y_train_current_cluster) == 0:
                    current_pyck = 0
                else:
                    current_pyck = len(Y_train_current_cluster[Y_train_current_cluster == value_current_class]) / \
                                   len(Y_train_current_cluster)
                self.pyck_[current_class][current_cluster] = current_pyck

        for i in range(self.minimum_time_stamp, X_train.shape[1] + 1):
            self.classifier_.append(MLPClassifier(solver=self.solver, hidden_layer_sizes=self.hidden_layer_sizes,
                                                  random_state=self.random_state,
                                                  max_iter=self.maximum_iteration))
            self.classifier_[-1].fit(X_train1[:, :i], Y_train1)
            for current_cluster in range(0, self.number_cluster):
                index_cluster = np.where(c_k2 == current_cluster)
                if len(index_cluster[0]) != 0:
                    X_train2_current_cluster = np.squeeze(X_train2[index_cluster, :i], axis=0)
                    Y_train2_current_cluster = Y_train2[tuple(index_cluster)]
                    Y_train2_hat = self.classifier_[-1].predict(X_train2_current_cluster[:, :i])
                    self.clusters_["ck_cm{0}".format(current_cluster)].append(confusion_matrix(Y_train2_current_cluster,
                                                                                               Y_train2_hat,
                                                                                               labels=classes_y))
                    column_sum = self.clusters_["ck_cm{0}".format(current_cluster)][-1].sum(1)
                    for current_class in range(0, self.number_classes_):
                        if column_sum[current_class] == 0:
                            column_sum[current_class] = 1
                    matrix_proba_current_cluster_class = \
                        np.asarray([[self.clusters_["ck_cm{0}".format(current_cluster)][-1][0][0] / column_sum[0],
                                     self.clusters_["ck_cm{0}".format(current_cluster)][-1][0][1] / column_sum[0]],
                                    [self.clusters_["ck_cm{0}".format(current_cluster)][-1][1][0] / column_sum[1],
                                     self.clusters_["ck_cm{0}".format(current_cluster)][-1][1][1] / column_sum[1]]])
                    self.pyhatyck_["pyhatycks{0}".format(current_cluster)].append(matrix_proba_current_cluster_class)

    def get_avg_dist(self, truncated_xt):
        """
        Compute the average euclidean distance of a given time series to the centroid of each cluster of the model
        :param xt: a time series that may  be incomplete
        :return: the average euclidean distance of the time series to the centroid of each cluster of the model
        """

        sum_distances = 0
        minimum_distance = 0
        for current_cluster in range(0, self.number_cluster):
            euclidean_distance = np.linalg.norm(truncated_xt - self.cluster_.cluster_centers_[current_cluster, :len(truncated_xt)])
            sum_distances = sum_distances + euclidean_distance
            if minimum_distance < euclidean_distance:
                minimum_distance = euclidean_distance
        return sum_distances / self.number_cluster, minimum_distance

    def get_dist(self, truncated_xt, wanted_cluster):
        """
        Compute the euclidean distance between a time series xt and the centroid of a cluster of the model
        :param xt: a time series that may be incomplete
        :param wanted_cluster: a figure that represent a cluster for which the distance is wanted
        :return: the expected euclidean distance
        """

        euclidean_dist = np.linalg.norm(truncated_xt - self.cluster_.cluster_centers_[wanted_cluster, :len(truncated_xt)])

        return euclidean_dist

    def get_cluster_prob(self, truncated_xt):
        """
        This function computes the probabilities of the time series xt to be in a cluster of the model
        :param xt: a time series that may be incomplete
        :return: the probabilities of the given time series to be in the clusters of the model
        """
        average_distance, minimum_distance = self.get_avg_dist(truncated_xt)
        minimum_distance_to_average = (average_distance - minimum_distance) / average_distance
        sum_sk = 0
        sk = []
        for current_cluster in range(0, self.number_cluster):
            d_k = self.get_dist(truncated_xt, current_cluster)
            Delta_k = (average_distance - d_k) / average_distance
            proba_current_cluster = 1 / (math.exp(self.lamb * minimum_distance_to_average) + math.exp(self.lamb * (minimum_distance_to_average - Delta_k)) + 10e-6)
            sk.append(proba_current_cluster)
            sum_sk = sum_sk + sk[-1]
        final_sk = [(x / sum_sk) for x in sk]
        return final_sk

    def cost_function(self, truncated_xt, tau):
        """
        From a incomplete series xt, this function compute the expected cost of a prediction made at time "last time of
        xt + tau"
        :param xt: a time series that may be incomplete for which we want to have the cost
        :param tau: gives the future time for which the cost is calculated
        :return: the computed cost
        """
        cost = 0
        proba_clusters = self.get_cluster_prob(truncated_xt=truncated_xt)
        truncated_t = truncated_xt.shape[-1]
        for current_cluster in range(0, self.number_cluster):
            cost_y = 0
            for y in range(0, self.number_classes_):
                cost_y_hat = 0
                for y_hat in range(0, self.number_classes_):
                    if y != y_hat:
                        cost_y_hat = cost_y_hat + self.pyhatyck_["pyhatycks{0}".format(current_cluster)][
                            truncated_t + tau - self.minimum_time_stamp][y][y_hat]
                cost_y = cost_y + self.pyck_[y, current_cluster] * cost_y_hat
            cost = cost + proba_clusters[current_cluster] * cost_y
        cost = cost + self.cost_time(t=truncated_t + tau)

        return cost

    @staticmethod
    def minimize_integer(end_of_time, function_to_minimize, xt):
        """
        We want to minimize a function "funct" according a time series "xt" for a list of integers from 0 to "stop"
        :param end_of_time: the highest integer at which the function to be minimized is calculated
        :param function_to_minimize: the function of interest
        :param xt: the time series related to the function "funct"
        :return: The integer in {0,...,stop} that minimizes "funct"
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
        :param xt: a time series that is probably incomplete but that nonetheless we want to classify
        :param cost_function: The cost function that will be use to achieve our classification purpose
        :return: The predicted classes and the time at which each of them has been predicted in two separate vectors
        """
        time_stamp = xt.shape[0]
        xt = np.reshape(xt, (1, time_stamp))
        time_prediction = self.minimum_time_stamp
        stop_criterion = False
        while stop_criterion is False:
            minimum_tau, minimum_loss = self.minimize_integer(end_of_time=time_stamp - time_prediction,
                                                              function_to_minimize=cost_function,
                                                              xt=xt[:, :time_prediction])
            if time_prediction == time_stamp:
                result = self.classifier_[-1].predict(xt)
                result_proba = self.classifier_[-1].predict_proba(xt)
                stop_criterion = True
                minimum_tau, minimum_loss = self.minimize_integer(end_of_time=time_stamp - time_prediction,
                                                                  function_to_minimize=cost_function,
                                                                  xt=xt[:, :time_prediction])
                loss_exit = minimum_loss
            elif minimum_tau == time_prediction:
                result = self.classifier_[time_prediction - self.minimum_time_stamp].predict(xt[:, :time_prediction])
                result_proba = self.classifier_[time_prediction -
                                                self.minimum_time_stamp].predict_proba(xt[:, :time_prediction])
                loss_exit = minimum_loss
                stop_criterion = True
            else:
                time_prediction = time_prediction + 1
        return result, time_prediction, result_proba, loss_exit

    @staticmethod
    def avg_prediction(time_predictions):
        """
        Computes the average value of the given vector
        :param time_predictions: The times at which the prediction have been made from the function "predict"
        :return: The average value of the input vector
        >>> model = NonMyopic(number_cluster=3, solver="sgd", hidden_layer_sizes=26, random_state=1, maximum_iteration=300, minimum_time_stamp=4, lamb=10)
        >>> model.avg_prediction([4, 5, 6, 5])
        5.0
        """
        avgerage_time = sum(time_predictions)/len(time_predictions)
        variance_time = np.var(time_predictions)
        return avgerage_time, variance_time

    @staticmethod
    def accuracy_prediction(Y_test, Y_pred_test):
        """
        Computes the accuracy of a vector of classes' prediction using the true classes
        :param Y_test: The true classes of the test dataset
        :param Y_pred_test: The predicted classes for the function "predict"
        :return: The accuracy of the predicted classes, that is, the part of correctly predicted series amongst all
        series
        >>> model = NonMyopic(number_cluster=3, solver="sgd", hidden_layer_sizes=26, random_state=1, maximum_iteration=300, minimum_time_stamp=4, lamb=10)
        >>> model.accuracy_prediction([-1, 1, 1, -1], [-1, 1, -1, 1])
        0.5
        """
        good_prediction = 0
        for class_index in range(0, len(Y_test)):
            if Y_test[class_index] == Y_pred_test[class_index]:
                good_prediction = good_prediction + 1
        return good_prediction / len(Y_test)

    def predict(self, X_test):
        """
                Predicts a test dataset of time series in an early fashion
                :param X_test: The data set that needs to be predict
                :param kwargs_up: The arguments that are needed for the cost function
                :return: A vector of length len(X_test) containing the classes predicted for the data set and another
                vector of
                length len(X_test) containing the time at which the predictions hae been made
                """
        Y_pred_test = []

        for t in range(0, X_test.shape[0]):
            new_classif, new_time, new_proba, new_cost = self.classification(X_test[t], self.cost_function)
            Y_pred_test.append(new_classif)
        return Y_pred_test

    def predict_completed(self, X_test, Y_test):
        """
        Predicts a test dataset of time series in an early fashion
        :param X_test: The data set that needs to be predict
        :param kwargs_up: The arguments that are needed for the cost function
        :return: A vector of length len(X_test) containing the classes predicted for the data set and another vector of
        length len(X_test) containing the time at which the predictions hae been made
        """
        Y_pred_test = []
        time_predictions = []
        pred_proba = []
        cost_average = 0

        for current_serie in range(0, X_test.shape[0]):
            new_classif, new_time, new_proba, new_cost = self.classification(X_test[current_serie], self.cost_function)
            Y_pred_test.append(new_classif)
            time_predictions.append(new_time)
            pred_proba.append(new_proba)
            cost_average = cost_average + new_cost
        average_pred, variance_pred = self.avg_prediction(time_predictions)
        pred_proba = np.reshape(np.asarray(pred_proba), (X_test.shape[0], 2))
        auc = roc_auc_score(Y_test, pred_proba[:, 1])
        cost_average = cost_average / X_test.shape[0]
        return average_pred, variance_pred, auc, cost_average

    def graphic(self, PATH, name, prediction_time=None):
        plt.clf()
        for current_cluster in range(0, self.number_cluster):
            for current_class in range(0, self.number_classes_):
                for current_hat_class in range(0, self.number_classes_):
                    if current_class != current_hat_class:
                        sum_prob = [self.pyck_[current_class][current_cluster] * item[current_class][current_hat_class]
                                    for item in self.pyhatyck_["pyhatycks{0}".format(current_cluster)]]
                        lab = \
                            "y=" + str(current_class) + "yhat=" + str(current_hat_class) + "ck=" + str(current_cluster)
                        plt.plot(sum_prob, label=lab)
        plt.legend()
        if prediction_time is not None:
            plt.title("The prediction time is " + str(prediction_time))
        plt.savefig(PATH + name + ".png", dpi=500)

    def cost_time(self, t):
        time_stamp = np.linspace(0, self.len_X_, num=self.len_X_)
        return time_stamp[t] * self.cost_time_parameter


if __name__ == '__main__':

    doctesting = False
    if doctesting:
        import doctest
        doctest.testmod()
    cost_time_parameter = args.cost_time_parameter
    PATH = "/home/adr2.local/painblanc_f/Téléchargements/Base de données/UCRArchive_2018/TwoLeadECG/"
    X_train = np.genfromtxt(PATH + "X_train.csv", delimiter=",")
    Y_train = np.genfromtxt(PATH + "Y_train.csv", delimiter=",")
    X_test = np.genfromtxt(PATH + "X_test.csv", delimiter=",")
    Y_test = np.genfromtxt(PATH + "Y_test.csv", delimiter=",")

    #Cross-validation part
    model = NonMyopic(number_cluster=3, solver="sgd", hidden_layer_sizes=41, random_state=1, maximum_iteration=10,
                      minimum_time_stamp=78, lamb=10, cost_time_parameter=cost_time_parameter)
    parameters = {"number_cluster": [2, 3, 4], "lamb": [1, 10]}
    CV = GridSearchCV(model, param_grid=parameters, cv=3)
    CV.fit(X_train, Y_train)
    CV.best_estimator_.predict_completed(X_test, Y_test)
