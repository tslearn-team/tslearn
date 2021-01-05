from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import coo_matrix

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..utils import to_time_series_dataset, check_dims
from ..neighbors import KNeighborsTimeSeriesClassifier
from ..clustering import TimeSeriesKMeans
from ..bases import TimeSeriesBaseEstimator


class NonMyopicEarlyClassifier(ClassifierMixin, TimeSeriesBaseEstimator):
    """Early Classification modelling for time series using the model
    presented in [1]_.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.

    base_classifier : Estimator or None
        Estimator (instance) to be cloned and used for classifications.
        If None, the chosen classifier is a 1NN with Euclidean metric.

    min_t : int
        Earliest time at which a classification can be performed on a time
        series

    lamb : float
        Value of the hyper parameter lambda used during the computation of the
        cost function to evaluate the probability
        that a time series belongs to a cluster given the time series.

    cost_time_parameter : float
        Parameter of the cost function of time. This function is of the form :
        f(time) = time * cost_time_parameter

    random_state: int
        Random state of the base estimator

    Attributes
    --------------------

    classifiers_ : list
        A list containing all the classifiers trained for the model, that is,
        (maximum_time_stamp - min_t) elements.

    pyhatyck_ : array like of shape (maximum_time_stamp - min_t, n_cluster, __n_classes, __n_classes)
        Contains the probabilities of being classified as class y_hat given
        class y and cluster ck for a trained classifier. The penultimate
        dimension of the array is associated to the true
        class of the series and the last dimension to the predicted class.


    pyck_ : array like of shape (__n_classes, n_cluster)
        Contains the probabilities of being of true class y given a cluster ck

    X_fit_dims : tuple of the same shape as the training dataset


    Examples
    --------
    >>> dataset = to_time_series_dataset([[1, 2, 3, 4, 5, 6],
    ...                                   [1, 2, 3, 4, 5, 6],
    ...                                   [1, 2, 3, 4, 5, 6],
    ...                                   [1, 2, 3, 3, 2, 1],
    ...                                   [1, 2, 3, 3, 2, 1],
    ...                                   [1, 2, 3, 3, 2, 1],
    ...                                   [3, 2, 1, 1, 2, 3],
    ...                                   [3, 2, 1, 1, 2, 3]])
    >>> y = [0, 0, 0, 1, 1, 1, 0, 0]
    >>> model = NonMyopicEarlyClassifier(n_clusters=3, lamb=1000.,
    ...                                  cost_time_parameter=.1,
    ...                                  random_state=0)
    >>> model.fit(dataset, y)  # doctest: +ELLIPSIS
    NonMyopicEarlyClassifier(...)
    >>> print(type(model.classifiers_))
    <class 'dict'>
    >>> print(model.pyck_)
    [[0. 1. 1.]
     [1. 0. 0.]]
    >>> preds, pred_times = model.predict_class_and_earliness(dataset)
    >>> preds
    array([0, 0, 0, 1, 1, 1, 0, 0])
    >>> pred_times
    array([4, 4, 4, 4, 4, 4, 1, 1])
    >>> pred_probas, pred_times = model.predict_proba_and_earliness(dataset)
    >>> pred_probas
    array([[1., 0.],
           [1., 0.],
           [1., 0.],
           [0., 1.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [1., 0.]])
    >>> pred_times
    array([4, 4, 4, 4, 4, 4, 1, 1])

    References
    ----------
    .. [1] A. Dachraoui, A. Bondu & A. Cornuejols. Early classification of time
       series as a non myopic sequential decision making problem.
       ECML/PKDD 2015
    """

    def __init__(self, n_clusters=2, base_classifier=None,
                 min_t=1, lamb=1., cost_time_parameter=1., random_state=None):
        super(NonMyopicEarlyClassifier, self).__init__()
        self.base_classifier = base_classifier
        self.n_clusters = n_clusters
        self.min_t = min_t
        self.lamb = lamb
        self.cost_time_parameter = cost_time_parameter
        self.random_state = random_state

    @property
    def classes_(self):
        if hasattr(self, 'classifiers_'):
            return self.classifiers_[self.min_t].classes_
        else:
            return None

    def fit(self, X, y):
        """
        Fit early classifier.

        Parameters
        ----------
        X : array-like of shape (n_series, n_timestamps, n_features)
            Training data, where `n_series` is the number of time series,
            `n_timestamps` is the number of timestamps in the series
            and `n_features` is the number of features recorded at each
            timestamp.

        y : array-like of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self : returns an instance of self.
        """

        X = check_array(X, allow_nd=True)
        X = check_dims(X)
        X = to_time_series_dataset(X)
        y_arr = np.array(y)
        label_set = np.unique(y_arr)

        self.cluster_ = TimeSeriesKMeans(n_clusters=self.n_clusters,
                                         random_state=self.random_state)
        if self.base_classifier is not None:
            clf = self.base_classifier
        else:
            clf = KNeighborsTimeSeriesClassifier(n_neighbors=1,
                                                 metric="euclidean")
        self.__n_classes_ = len(label_set)
        self._X_fit_dims = X.shape
        sz = X.shape[1]
        self.classifiers_ = {t: clone(clf)
                             for t in range(self.min_t, sz + 1)}
        self.pyhatyck_ = np.empty((sz - self.min_t + 1,
                                   self.n_clusters,
                                   self.__n_classes_, self.__n_classes_))
        c_k = self.cluster_.fit_predict(X)
        X1, X2, c_k1, c_k2, y1, y2 = train_test_split(
            X, c_k, y_arr,
            test_size=0.5,
            stratify=c_k,
            random_state=self.random_state
        )

        label_to_ind = {lab: ind for ind, lab in enumerate(label_set)}
        y_ = np.array([label_to_ind.get(lab, self.__n_classes_ + 1)
                       for lab in y_arr])

        vector_of_ones = np.ones((X.shape[0], ))
        self.pyck_ = coo_matrix(
            (vector_of_ones, (y_, c_k)),
            shape=(self.__n_classes_, self.n_clusters),
        ).toarray()
        self.pyck_ /= self.pyck_.sum(axis=0, keepdims=True)
        for t in range(self.min_t, sz + 1):
            self.classifiers_[t].fit(X1[:, :t], y1)
            for k in range(0, self.n_clusters):
                index = (c_k2 == k)
                if index.shape[0] != 0:
                    X2_current_cluster = X2[index, :t]
                    y2_current_cluster = y2[index]
                    y2_hat = self.classifiers_[t].predict(
                        X2_current_cluster[:, :t]
                    )
                    conf_matrix = confusion_matrix(y2_current_cluster, y2_hat,
                                                   labels=label_set)
                    # normalize parameter seems to be quite recent in sklearn,
                    # so let's do it ourselves
                    normalizer = conf_matrix.sum(axis=0, keepdims=True)
                    normalizer[normalizer == 0] = 1  # Avoid divide by 0
                    conf_matrix = conf_matrix / normalizer

                    # pyhatyck_ stores
                    # P_{t+\tau}(\hat{y} | y, c_k) \delta_{y \neq \hat{y}}
                    # elements so it should have a null diagonal because of
                    # the \delta_{y \neq \hat{y}} term
                    np.fill_diagonal(conf_matrix, 0)
                    self.pyhatyck_[t - self.min_t, k] = conf_matrix
        return self

    def get_cluster_probas(self, Xi):
        r"""Compute cluster probability :math:`P(c_k | Xi)`.

        This quantity is computed using the following formula:

        .. math::

            P(c_k | Xi) = \frac{s_k(Xi)}{\sum_j s_j(Xi)}

        where

        .. math::

            s_k(Xi) = \frac{1}{1 + \exp{-\lambda \Delta_k(Xi)}}

        with

        .. math::

            \Delta_k(Xi) = \frac{\bar{D} - d(Xi, c_k)}{\bar{D}}

        and :math:`\bar{D}` is the average of the distances between `Xi` and
        the cluster centers.

        Parameters
        ----------
        Xi: numpy array, shape (t, d)
            A time series observed up to time t

        Returns
        -------
        probas : numpy array, shape (n_clusters, )

        Examples
        --------
        >>> from tslearn.utils import to_time_series
        >>> dataset = to_time_series_dataset([[1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [3, 2, 1, 1, 2, 3],
        ...                                   [3, 2, 1, 1, 2, 3]])
        >>> y = [0, 0, 0, 1, 1, 1, 0, 0]
        >>> ts0 = to_time_series([1, 2])
        >>> model = NonMyopicEarlyClassifier(n_clusters=3, lamb=0.,
        ...                                  random_state=0)
        >>> probas = model.fit(dataset, y).get_cluster_probas(ts0)
        >>> probas.shape
        (3,)
        >>> probas  # doctest: +ELLIPSIS
        array([0.33..., 0.33..., 0.33...])
        >>> model = NonMyopicEarlyClassifier(n_clusters=3, lamb=10000.,
        ...                                  random_state=0)
        >>> probas = model.fit(dataset, y).get_cluster_probas(ts0)
        >>> probas.shape
        (3,)
        >>> probas
        array([0.5, 0.5, 0. ])
        >>> ts1 = to_time_series([3, 2])
        >>> model.get_cluster_probas(ts1)
        array([0., 0., 1.])
        """
        Xi = check_array(Xi)
        diffs = Xi[np.newaxis, :] - self.cluster_.cluster_centers_[:, :len(Xi)]
        distances_clusters = np.linalg.norm(diffs, axis=(1, 2))
        average_distance = np.mean(distances_clusters)
        delta_k = 1. - distances_clusters / average_distance
        s_k = 1. / (1. + np.exp(-self.lamb * delta_k))
        return s_k / s_k.sum()

    def _expected_costs(self, Xi):
        r"""Compute expected future costs from an incoming time series `Xi`.

        This cost is computed, for a time horizon :math:`\tau`, as:

        .. math::

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

        Examples
        --------
        >>> from tslearn.utils import to_time_series
        >>> dataset = to_time_series_dataset([[1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [3, 2, 1, 1, 2, 3],
        ...                                   [3, 2, 1, 1, 2, 3]])
        >>> y = [0, 0, 0, 1, 1, 1, 0, 0]
        >>> ts1 = to_time_series([3, 2])
        >>> model = NonMyopicEarlyClassifier(n_clusters=3, lamb=10000.,
        ...                                  cost_time_parameter=1.,
        ...                                  random_state=0)
        >>> costs = model.fit(dataset, y)._expected_costs(ts1)
        >>> costs.shape
        (5,)
        >>> costs  # doctest: +ELLIPSIS
        array([2., 3., 4., 5., 6.])
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
        return cost + self._cost_time(np.arange(truncated_t,
                                                self._X_fit_dims[1] + 1))

    def _get_prediction_time(self, Xi):
        """Compute optimal prediction time for the incoming time series Xi.
        """
        time_prediction = None
        for t in range(self.min_t, self._X_fit_dims[1] + 1):
            tau_star = np.argmin(self._expected_costs(Xi=Xi[:t]))
            if (t == self._X_fit_dims[1]) or (tau_star == 0):
                time_prediction = t
                break
        return time_prediction

    def _predict_single_series(self, Xi):
        """
        This function classifies a single time series xt

        Parameters
        ----------
        xt: vector
            a time series that is probably incomplete but that nonetheless we
            want to classify
        Returns
        -------
        int: the class which is predicted
        int : the time of the prediction
        float : the probability used for computing the cost
        float : the loss when classifying
        """
        t = self._get_prediction_time(Xi)
        pred = self.classifiers_[t].predict([Xi[:t]])[0]
        return pred, t

    def _predict_single_series_proba(self, Xi):
        """
        This function classifies a single time series xt

        Parameters
        ----------
        Xi: vector
            a time series that is probably incomplete but that nonetheless we
            want to classify
        Returns
        -------
        int: the class which is predicted
        int : the time of the prediction
        float : the probability used for computing the cost
        float : the loss when classifying
        """
        t = self._get_prediction_time(Xi)
        pred = self.classifiers_[t].predict_proba([Xi[:t]])[0]
        return pred, t

    def predict_class_and_earliness(self, X):
        """
        Provide predicted class as well as prediction timestamps.

        Prediction timestamps are timestamps at which a prediction is made in
        early classification setting.

        Parameters
        ----------
        X : array-like of shape (n_series, n_timestamps, n_features)
            Vector to be scored, where `n_series` is the number of time series,
            `n_timestamps` is the number of timestamps in the series
            and `n_features` is the number of features recorded at each
            timestamp.

        Returns
        -------
        array, shape (n_samples,)
            Predicted classes.
        array-like of shape (n_series, )
            Prediction timestamps.
        """

        X = check_array(X, allow_nd=True)
        check_is_fitted(self, '_X_fit_dims')
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        y_pred = []
        time_prediction = []
        for i in range(0, X.shape[0]):
            cl, t = self._predict_single_series(X[i])
            y_pred.append(cl)
            time_prediction.append(t)
        return np.array(y_pred), np.array(time_prediction)

    def predict(self, X):
        """
        Provide predicted class.

        Parameters
        ----------
        X : array-like of shape (n_series, n_timestamps, n_features)
            Vector to be scored, where `n_series` is the number of time series,
            `n_timestamps` is the number of timestamps in the series
            and `n_features` is the number of features recorded at each
            timestamp.

        Returns
        -------
        array, shape (n_samples,)
            Predicted classes.
        """
        return self.predict_class_and_earliness(X)[0]

    def predict_proba_and_earliness(self, X):
        """
        Provide probability estimates as well as prediction timestamps.

        Prediction timestamps are timestamps at which a prediction is made in
        early classification setting.
        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_series, n_timestamps, n_features)
            Vector to be scored, where `n_series` is the number of time series,
            `n_timestamps` is the number of timestamps in the series
            and `n_features` is the number of features recorded at each
            timestamp.

        Returns
        -------
        array-like of shape (n_series, n_classes)
            Probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        array-like of shape (n_series, )
            Prediction timestamps.
        """

        X = check_array(X, allow_nd=True)
        check_is_fitted(self, '_X_fit_dims')
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        y_pred = []
        time_prediction = []
        for i in range(0, X.shape[0]):
            probas, t = self._predict_single_series_proba(X[i])
            y_pred.append(probas)
            time_prediction.append(t)
        return np.array(y_pred), np.array(time_prediction)

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_series, n_timestamps, n_features)
            Vector to be scored, where `n_series` is the number of time series,
            `n_timestamps` is the number of timestamps in the series
            and `n_features` is the number of features recorded at each
            timestamp.

        Returns
        -------
        array-like of shape (n_series, n_classes)
            Probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        return self.predict_proba_and_earliness(X)[0]

    def _cost_time(self, t):
        return t * self.cost_time_parameter

    def early_classification_cost(self, X, y):
        r"""
        Compute early classification score.

        The score is computed as:

        .. math::

            1 - acc + \alpha \frac{1}{n} \sum_i t_i

        where :math:`\alpha` is the trade-off parameter
        (`self.cost_time_parameter`) and :math:`t_i` are prediction timestamps.

        Parameters
        ----------
        X : array-like of shape (n_series, n_timestamps, n_features)
            Vector to be scored, where `n_series` is the number of time series,
            `n_timestamps` is the number of timestamps in the series
            and `n_features` is the number of features recorded at each
            timestamp.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------
        float
            Early classification cost (a positive number, the lower the better)

        Examples
        --------
        >>> dataset = to_time_series_dataset([[1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 4, 5, 6],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [1, 2, 3, 3, 2, 1],
        ...                                   [3, 2, 1, 1, 2, 3],
        ...                                   [3, 2, 1, 1, 2, 3]])
        >>> y = [0, 0, 0, 1, 1, 1, 0, 0]
        >>> model = NonMyopicEarlyClassifier(n_clusters=3, lamb=1000.,
        ...                                  cost_time_parameter=.1,
        ...                                  random_state=0)
        >>> model.fit(dataset, y)  # doctest: +ELLIPSIS
        NonMyopicEarlyClassifier(...)
        >>> preds, pred_times = model.predict_class_and_earliness(dataset)
        >>> preds
        array([0, 0, 0, 1, 1, 1, 0, 0])
        >>> pred_times
        array([4, 4, 4, 4, 4, 4, 1, 1])
        >>> model.early_classification_cost(dataset, y)
        0.325
        """
        y_pred, pred_times = self.predict_class_and_earliness(X)
        acc = accuracy_score(y, y_pred)
        return (1. - acc) + np.mean(self._cost_time(pred_times))

    def _more_tags(self):
        # Because some of the data validation checks rely on datasets that are
        # too small to pass here (only 1 item in one of the clusters, hence no
        # stratified split possible)
        return {"no_validation": True}
