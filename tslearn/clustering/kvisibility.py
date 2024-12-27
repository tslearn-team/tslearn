from sklearn.base import ClusterMixin

from sklearn.utils import check_random_state
import numpy


from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset, check_dims
from tslearn.metrics import cdist_normalized_cc, y_shifted_sbd_vec
from tslearn.bases import BaseModelPackage, TimeSeriesBaseEstimator

from .utils import (TimeSeriesCentroidBasedClusteringMixin,
                    _check_no_empty_cluster, _compute_inertia,
                    _check_initial_guess, EmptyClusterError)

__author__ = 'Sergio Iglesias_Perez seigpe[at]gmail.com'


class KVisibility(ClusterMixin, TimeSeriesCentroidBasedClusteringMixin,
             BaseModelPackage, TimeSeriesBaseEstimator):
    """KVisibility clustering for time series.

    KVisibility was originally presented in [1]_.

    Parameters
    ----------
    n_clusters : int (default: 3)
        Number of clusters to form.

    max_iter : int (default: 100)
        Maximum number of iterations of the k-Shape algorithm.

    tol : float (default: 1e-6)
        Inertia variation threshold. If at some point, inertia varies less than
        this threshold between two consecutive
        iterations, the model is considered to have converged and the algorithm
        stops.

    n_init : int (default: 1)
        Number of time the kmeans algorithm will be run with different
        centroid seeds. The final results will be the
        best output of n_init consecutive runs in terms of inertia.

    verbose : bool (default: False)
        Whether or not to print information about the inertia while learning
        the model.

    random_state : integer or numpy.RandomState, optional
        Generator used to initialize the centers. If an integer is given, it
        fixes the seed. Defaults to the global
        numpy random number generator.

    init : {'random' or ndarray} (default: 'random')
        Method for initialization.
        'random': choose k observations (rows) at random from data for the
        initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, ts_size, d)
        and gives the initial centers.

    Attributes
    ----------
    cluster_centers_ : numpy.ndarray of shape (sz, d).
        Centroids

    labels_ : numpy.ndarray of integers with shape (n_ts, ).
        Labels of each point

    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    n_iter_ : int
        The number of iterations performed during fit.

    Notes
    -----
        This method requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walks
    >>> X = random_walks(n_ts=50, sz=32, d=1)
    >>> kv = KVisibility(n_clusters=3, n_init=1, random_state=0).fit_predict(X)
    
    References
    ----------
    .. [1] J. Paparrizos & L. Gravano. k-Shape: Efficient and Accurate
       Clustering of Time Series. SIGMOD 2015. pp. 1855-1870.
    """

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-6, n_init=1,
                 verbose=False, random_state=None, init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.verbose = verbose
        self.init = init

    def _is_fitted(self):
        """
        Check if the model has been fit.

        Returns
        -------
        bool
        """

        check_is_fitted(self,
                        ['cluster_centers_', 'norms_', 'norms_centroids_'])
        return True

    def _ts_to_graph(self, X):
        ts_clustering = []
        ts_attr = []


        for ts in series:
            # ts for each time series
            g = HorizontalVG()
            g.build(ts)
            nx_g = g.as_networkx()

            density_h = nx.density(nx_g)
            max_grade_h = max(nx_g.degree, key=lambda x: x[1])[1]

            ################# Natural VG
            gn = NaturalVG()
            gn.build(ts)
            nx_gn = gn.as_networkx()
            density_n = nx.density(nx_gn)
            max_grade_n = max(nx_gn.degree, key=lambda x: x[1])[1]

            ts_attr.append([density_h, max_grade_h, density_n, max_grade_n])
        df = pd.DataFrame(ts_attr, columns=['density_h','max_degree_h','density_n','max_degree_n'])
        from sklearn.cluster import KMeans
        ts_features = np.array(df[['density_h','max_degree_h','density_n','max_degree_n']])
        return ts_features


    def fit(self, X, y=None):
        """Compute k-Shape clustering.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        y
            Ignored
        """
        X = check_array(X, allow_nd=True)

        max_attempts = max(self.n_init, 10)

        self.kmeans = None
        
        self.ts_features =  self._ts_to_graph(X)

        kmeans = KMeans(init="k-means++", n_clusters=num_cluster, n_init=4)
        kmeans.fit(self.ts_features)
        self.kmeans = kmeans
        return self

    def fit_predict(self, X, y=None):
        """Fit k-Shape clustering using X and then predict the closest cluster
        each time series in X belongs to.

        It is more efficient to use this method than to sequentially call fit
        and predict.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        y
            Ignored

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        
        self.ts_features =  self._ts_to_graph(X)

        kmeans = KMeans(init="k-means++", n_clusters=num_cluster, n_init=4)
        kmeans.fit(self.ts_features)
        self.kmeans = kmeans
        return self.kmeans.predict(self.ts_features)

    def predict(self, X):
        """Predict the closest cluster each time series in X belongs to.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset to predict.

        Returns
        -------
        labels : array of shape=(n_ts, )
            Index of the cluster each sample belongs to.
        """
        X = check_array(X, allow_nd=True)
        check_is_fitted(self,
                        ['cluster_centers_', 'norms_', 'norms_centroids_'])

        
        self.ts_features =  self._ts_to_graph(X)
        return self.kmeans.predict(self.ts_features)
