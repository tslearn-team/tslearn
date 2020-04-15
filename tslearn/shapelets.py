"""
The :mod:`tslearn.shapelets` module gathers Shapelet-based algorithms.

It depends on the `keras` library for optimization.
"""

from keras.models import Model
from keras.layers import Dense, Conv1D, Layer, Input, concatenate, add
from keras.metrics import (categorical_accuracy, categorical_crossentropy,
                           binary_accuracy, binary_crossentropy)
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from keras.regularizers import l2
from keras.initializers import Initializer
import keras.backend as K
from keras.engine import InputSpec
import numpy
try:
    from tensorflow.compat.v1 import set_random_seed
except ImportError:
    from tensorflow import set_random_seed

import warnings

from tslearn.utils import to_time_series_dataset, check_dims
from tslearn.clustering import TimeSeriesKMeans

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class GlobalMinPooling1D(Layer):
    """Global min pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    """

    def __init__(self, **kwargs):
        super(GlobalMinPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        return K.min(inputs, axis=1)


class GlobalArgminPooling1D(Layer):
    """Global min pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    """

    def __init__(self, **kwargs):
        super(GlobalArgminPooling1D, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        return K.cast(K.argmin(inputs, axis=1), dtype=K.floatx())


def _kmeans_init_shapelets(X, n_shapelets, shp_len, n_draw=10000):
    n_ts, sz, d = X.shape
    indices_ts = numpy.random.choice(n_ts, size=n_draw, replace=True)
    indices_time = numpy.random.choice(sz - shp_len + 1, size=n_draw,
                                       replace=True)
    subseries = numpy.zeros((n_draw, shp_len, d))
    for i in range(n_draw):
        subseries[i] = X[indices_ts[i],
                         indices_time[i]:indices_time[i] + shp_len]
    return TimeSeriesKMeans(n_clusters=n_shapelets,
                            metric="euclidean",
                            verbose=False).fit(subseries).cluster_centers_


class KMeansShapeletInitializer(Initializer):
    """Initializer that generates shapelet tensors based on a clustering of
    time series snippets.

    # Arguments
        dataset: a dataset of time series.
    """
    def __init__(self, X):
        self.X_ = to_time_series_dataset(X)

    def __call__(self, shape, dtype=None):
        n_shapelets, shp_len = shape
        shapelets = _kmeans_init_shapelets(self.X_,
                                           n_shapelets,
                                           shp_len)[:, :, 0]
        return K.tensorflow_backend._to_tensor(x=shapelets, dtype=K.floatx())

    def get_config(self):
        return {'data': self.X_}


class LocalSquaredDistanceLayer(Layer):
    """Pairwise (squared) distance computation between local patches and
    shapelets

    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        3D tensor with shape:
        `(batch_size, steps, n_shapelets)`
    """
    def __init__(self, n_shapelets, X=None, **kwargs):
        self.n_shapelets = n_shapelets
        if X is None or K.backend() != "tensorflow":
            self.initializer = "uniform"
        else:
            self.initializer = KMeansShapeletInitializer(X)  # TODO: v2-compatible initializer
        super(LocalSquaredDistanceLayer, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.n_shapelets, input_shape[2]),
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalSquaredDistanceLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        # (x - y)^2 = x^2 + y^2 - 2 * x * y
        x_sq = K.expand_dims(K.sum(x ** 2, axis=2), axis=-1)
        y_sq = K.reshape(K.sum(self.kernel ** 2, axis=1),
                         (1, 1, self.n_shapelets))
        xy = K.dot(x, K.transpose(self.kernel))
        return (x_sq + y_sq - 2 * xy) / K.int_shape(self.kernel)[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.n_shapelets

    def get_config(self):
        config = {'n_shapelets': self.n_shapelets}
        base_config = super(LocalSquaredDistanceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def grabocka_params_to_shapelet_size_dict(n_ts, ts_sz, n_classes, l, r):
    """Compute number and length of shapelets.

     This function uses the heuristic from [1]_.

    Parameters
    ----------
    n_ts: int
        Number of time series in the dataset
    ts_sz: int
        Length of time series in the dataset
    n_classes: int
        Number of classes in the dataset
    l: float
        Fraction of the length of time series to be used for base shapelet
        length
    r: int
        Number of different shapelet lengths to use

    Returns
    -------
    dict
        Dictionnary giving, for each shapelet length, the number of such
        shapelets to be generated

    Examples
    --------
    >>> d = grabocka_params_to_shapelet_size_dict(
    ...         n_ts=100, ts_sz=100, n_classes=3, l=0.1, r=2)
    >>> keys = sorted(d.keys())
    >>> print(keys)
    [10, 20]
    >>> print([d[k] for k in keys])
    [4, 4]


    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    base_size = int(l * ts_sz)
    base_size = max(base_size, 1)
    r = min(r, ts_sz)
    d = {}
    for sz_idx in range(r):
        shp_sz = base_size * (sz_idx + 1)
        n_shapelets = int(numpy.log10(n_ts *
                                      (ts_sz - shp_sz + 1) *
                                      (n_classes - 1)))
        n_shapelets = max(1, n_shapelets)
        d[shp_sz] = n_shapelets
    return d


class ShapeletModel(BaseEstimator, ClassifierMixin, TransformerMixin):
    r"""Learning Time-Series Shapelets model.


    Learning Time-Series Shapelets was originally presented in [1]_.

    From an input (possibly multidimensional) time series :math:`x` and a set
    of shapelets :math:`\{s_i\}_i`, the :math:`i`-th coordinate of the Shapelet
    transform is computed as:

    .. math::

        ST(x, s_i) = \min_t \sum_{\delta_t}
            \left\|x(t+\delta_t) - s_i(\delta_t)\right\|_2^2

    The Shapelet model consists in a logistic regression layer on top of this
    transform. Shapelet coefficients as well as logistic regression weights are
    optimized by gradient descent on a L2-penalized cross-entropy loss.

    Parameters
    ----------
    n_shapelets_per_size: dict (default: None)
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value)
    max_iter: int (default: 10,000)
        Number of training epochs.

        .. versionchanged:: 0.3
            default value for max_iter is set to 10,000 instead of 100

    batch_size: int (default: 256)
        Batch size to be used.
    verbose_level: {0, 1, 2} (default: 0)
        `keras` verbose level.

        .. deprecated:: 0.2
            verbose_level is deprecated in version 0.2 and will be
            removed in 0.4. Use `verbose` instead.

    verbose: {0, 1, 2} (default: 0)
        `keras` verbose level.
    optimizer: str or keras.optimizers.Optimizer (default: "sgd")
        `keras` optimizer to use for training.
    weight_regularizer: float or None (default: 0.)
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer. If 0, no regularization is performed.
    shapelet_length: float (default 0.15)
        The length of the shapelets, expressed as a fraction of the ts length
    total_lengths: int (default 3)
        The number of different shapelet lengths. Will extract shapelets of
        length i * shapelet_length for i in [1, total_lengths]
    random_state : int or None, optional (default: None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    shapelets_ : numpy.ndarray of objects, each object being a time series
        Set of time-series shapelets.

    shapelets_as_time_series_ : numpy.ndarray of shape (n_shapelets, sz_shp, d) where `sz_shp` is the maximum of all shapelet sizes
        Set of time-series shapelets formatted as a ``tslearn`` time series
        dataset.

    transformer_model_ : keras.Model
        Transforms an input dataset of timeseries into distances to the
        learned shapelets.

    locator_model_ : keras.Model
        Returns the indices where each of the shapelets can be found (minimal
        distance) within each of the timeseries of the input dataset.

    model_ : keras.Model
        Directly predicts the class probabilities for the input timeseries.

    Notes
    -----
        This implementation requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=16, d=2, n_blobs=3)
    >>> clf = ShapeletModel(n_shapelets_per_size={4: 5}, max_iter=1, verbose=0)
    >>> clf.fit(X, y).shapelets_.shape
    (5,)
    >>> clf.shapelets_[0].shape
    (4, 2)
    >>> clf.predict(X).shape
    (30,)
    >>> clf.predict_proba(X).shape
    (30, 3)
    >>> clf.transform(X).shape
    (30, 5)

    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    def __init__(self, n_shapelets_per_size=None,
                 max_iter=10000,
                 batch_size=256,
                 verbose=0,
                 verbose_level=None,
                 optimizer="sgd",
                 weight_regularizer=0.,
                 shapelet_length=0.15,
                 total_lengths=3,
                 random_state=None):
        self.n_shapelets_per_size = n_shapelets_per_size
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose_level = verbose_level
        self.verbose = verbose
        self.optimizer = optimizer
        self.weight_regularizer = weight_regularizer
        self.shapelet_length = shapelet_length
        self.total_lengths = total_lengths
        self.random_state = random_state

        if max_iter == 10000:
            warnings.warn("The default value of max_iter has changed "
                          "from 100 to 10000 starting from version 0.3 for "
                          "the model to be more likely to converge. "
                          "Explicitly set your max_iter value to "
                          "avoid this warning.", FutureWarning)

    @property
    def _n_shapelet_sizes(self):
        return len(self.n_shapelets_per_size)

    @property
    def shapelets_(self):
        total_n_shp = sum(self.n_shapelets_per_size.values())
        shapelets = numpy.empty((total_n_shp, ), dtype=object)
        idx = 0
        for i, shp_sz in enumerate(sorted(self.n_shapelets_per_size.keys())):
            n_shp = self.n_shapelets_per_size[shp_sz]
            for idx_shp in range(idx, idx + n_shp):
                shapelets[idx_shp] = numpy.zeros((shp_sz, self.d_))
            for di in range(self.d_):
                layer = self.model_.get_layer("shapelets_%d_%d" % (i, di))
                for inc, shp in enumerate(layer.get_weights()[0]):
                    shapelets[idx + inc][:, di] = shp
            idx += n_shp
        assert idx == total_n_shp
        return shapelets

    @property
    def shapelets_as_time_series_(self):
        """Set of time-series shapelets formatted as a ``tslearn`` time series
        dataset.

        Examples
        --------
        >>> from tslearn.generators import random_walk_blobs
        >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=256, d=1, n_blobs=3)
        >>> model = ShapeletModel(n_shapelets_per_size={3: 2, 4: 1},
        ...                       max_iter=1)
        >>> _ = model.fit(X, y)
        >>> model.shapelets_as_time_series_.shape
        (3, 4, 1)
        """
        total_n_shp = sum(self.n_shapelets_per_size.values())
        shp_sz = max(self.n_shapelets_per_size.keys())
        non_formatted_shapelets = self.shapelets_
        d = non_formatted_shapelets[0].shape[1]
        shapelets = numpy.zeros((total_n_shp, shp_sz, d)) + numpy.nan
        for i in range(total_n_shp):
            sz = non_formatted_shapelets[i].shape[0]
            shapelets[i, :sz, :] = non_formatted_shapelets[i]
        return shapelets

    def fit(self, X, y):
        """Learn time-series shapelets.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        y : array-like of shape=(n_ts, )
            Time series labels.
        """
        if self.verbose_level is not None:
            warnings.warn(
                "'verbose_level' is deprecated in version 0.2 and will be "
                "removed in 0.4. Use 'verbose' instead.",
                DeprecationWarning, stacklevel=2)
            self.verbose = self.verbose_level

        X, y = check_X_y(X, y, allow_nd=True)
        X = to_time_series_dataset(X)
        X = check_dims(X, X_fit=None)

        set_random_seed(seed=self.random_state)
        numpy.random.seed(seed=self.random_state)

        n_ts, sz, d = X.shape
        self._X_fit = X

        self.model_ = None
        self.transformer_model_ = None
        self.locator_model_ = None
        self.categorical_y_ = False
        self.label_binarizer_ = None
        self.d_ = d

        if y.ndim == 1 or y.shape[1] == 1:
            self.label_binarizer_ = LabelBinarizer().fit(y)
            y_ = self.label_binarizer_.transform(y)
            self.classes_ = self.label_binarizer_.classes_
        else:
            y_ = y
            self.categorical_y_ = True
            self.classes_ = numpy.unique(y)
            assert y_.shape[1] != 2, ("Binary classification case, " +
                                      "monodimensional y should be passed.")

        if y_.ndim == 1 or y_.shape[1] == 1:
            n_classes = 2
        else:
            n_classes = y_.shape[1]

        if self.n_shapelets_per_size is None:
            sizes = grabocka_params_to_shapelet_size_dict(n_ts, sz, n_classes,
                                                          self.shapelet_length,
                                                          self.total_lengths)
            self.n_shapelets_per_size_ = sizes
        else:
            self.n_shapelets_per_size_ = self.n_shapelets_per_size

        self._set_model_layers(X=X, ts_sz=sz, d=d, n_classes=n_classes)
        self.transformer_model_.compile(loss="mean_squared_error",
                                        optimizer=self.optimizer)
        self.locator_model_.compile(loss="mean_squared_error",
                                    optimizer=self.optimizer)
        self._set_weights_false_conv(d=d)
        self.model_.fit(
            [X[:, :, di].reshape((n_ts, sz, 1)) for di in range(d)], y_,
            batch_size=self.batch_size, epochs=self.max_iter,
            verbose=self.verbose
        )
        self.n_iter_ = len(self.model_.history.history)
        return self

    def predict(self, X):
        """Predict class for a given set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, ) or (n_ts, n_classes), depending on the shape
        of the label vector provided at training time.
            Index of the cluster each sample belongs to or class probability
            matrix, depending on what was provided at training time.
        """
        check_is_fitted(self, '_X_fit')
        X = check_array(X, allow_nd=True)
        X = to_time_series_dataset(X)
        X = check_dims(X, X_fit=self._X_fit)

        categorical_preds = self.predict_proba(X)
        if self.categorical_y_:
            return categorical_preds
        else:
            return self.label_binarizer_.inverse_transform(categorical_preds)

    def predict_proba(self, X):
        """Predict class probability for a given set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_classes),
            Class probability matrix.
        """
        check_is_fitted(self, '_X_fit')
        X = check_array(X, allow_nd=True)
        X = to_time_series_dataset(X)
        X = check_dims(X, self._X_fit)
        n_ts, sz, d = X.shape
        categorical_preds = self.model_.predict(
            [X[:, :, di].reshape((n_ts, sz, 1)) for di in range(self.d_)],
            batch_size=self.batch_size, verbose=self.verbose
        )

        if categorical_preds.shape[1] == 1 and len(self.classes_) == 2:
            categorical_preds = numpy.hstack((1 - categorical_preds,
                                              categorical_preds))

        return categorical_preds

    def transform(self, X):
        """Generate shapelet transform for a set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_shapelets)
            Shapelet-Transform of the provided time series.
        """
        check_is_fitted(self, '_X_fit')
        X = check_array(X, allow_nd=True)
        X = to_time_series_dataset(X)
        X = check_dims(X, X_fit=self._X_fit)
        n_ts, sz, d = X.shape
        pred = self.transformer_model_.predict(
            [X[:, :, di].reshape((n_ts, sz, 1)) for di in range(self.d_)],
            batch_size=self.batch_size, verbose=self.verbose
        )
        return pred

    def locate(self, X):
        """Compute shapelet match location for a set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_shapelets)
            Location of the shapelet matches for the provided time series.

        Examples
        --------
        >>> from tslearn.generators import random_walk_blobs
        >>> X = numpy.zeros((3, 10, 1))
        >>> X[0, 4:7, 0] = numpy.array([1, 2, 3])
        >>> y = [1, 0, 0]
        >>> # Data is all zeros except a motif 1-2-3 in the first time series
        >>> clf = ShapeletModel(n_shapelets_per_size={3: 1}, max_iter=0,
        ...                     verbose=0)
        >>> _ = clf.fit(X, y)
        >>> weights_shapelet = [
        ...     numpy.array([[1, 2, 3]])
        ... ]
        >>> clf.set_weights(weights_shapelet, layer_name="shapelets_0_0")
        >>> clf.locate(X)
        array([[4],
               [0],
               [0]])
        """
        X = check_dims(X, X_fit=self._X_fit)
        X = check_array(X, allow_nd=True)
        X = to_time_series_dataset(X)
        X = check_dims(X, X_fit=self._X_fit)
        n_ts, sz, d = X.shape
        locations = self.locator_model_.predict(
            [X[:, :, di].reshape((n_ts, sz, 1)) for di in range(self.d_)],
            batch_size=self.batch_size, verbose=self.verbose
        )
        return locations.astype(numpy.int)

    def _set_weights_false_conv(self, d):
        shapelet_sizes = sorted(self.n_shapelets_per_size_.keys())
        for i, sz in enumerate(shapelet_sizes):
            for di in range(d):
                layer = self.model_.get_layer("false_conv_%d_%d" % (i, di))
                layer.set_weights([numpy.eye(sz).reshape((sz, 1, sz))])

    def _set_model_layers(self, X, ts_sz, d, n_classes):
        inputs = [Input(shape=(ts_sz, 1),
                        name="input_%d" % di)
                  for di in range(d)]
        shapelet_sizes = sorted(self.n_shapelets_per_size_.keys())
        pool_layers = []
        pool_layers_locations = []
        for i, sz in enumerate(sorted(shapelet_sizes)):
            transformer_layers = [
                Conv1D(
                    filters=sz, kernel_size=sz,
                    trainable=False, use_bias=False,
                    name="false_conv_%d_%d" % (i, di)
                )(inputs[di]) for di in range(d)
            ]
            shapelet_layers = [
                LocalSquaredDistanceLayer(
                    self.n_shapelets_per_size_[sz], X=X,
                    name="shapelets_%d_%d" % (i, di)
                )(transformer_layers[di]) for di in range(d)
            ]

            if d == 1:
                sum_shap = shapelet_layers[0]
            else:
                sum_shap = add(shapelet_layers)

            gp = GlobalMinPooling1D(name="min_pooling_%d" % i)(sum_shap)
            gap = GlobalArgminPooling1D(name="min_pooling_%d" % i)(sum_shap)
            pool_layers.append(gp)
            pool_layers_locations.append(gap)
        if len(shapelet_sizes) > 1:
            concatenated_features = concatenate(pool_layers)
            concatenated_locations = concatenate(pool_layers_locations)
        else:
            concatenated_features = pool_layers[0]
            concatenated_locations = pool_layers_locations[0]

        if self.weight_regularizer > 0:
            regularizer = l2(self.weight_regularizer)
        else:
            regularizer = None

        if n_classes > 2:
            loss = "categorical_crossentropy"
            metrics = [categorical_accuracy, categorical_crossentropy]
        else:
            loss = "binary_crossentropy"
            metrics = [binary_accuracy, binary_crossentropy]

        outputs = Dense(units=n_classes if n_classes > 2 else 1,
                        activation="softmax" if n_classes > 2 else "sigmoid",
                        kernel_regularizer=regularizer,
                        name="classification")(concatenated_features)
        self.model_ = Model(inputs=inputs, outputs=outputs)
        self.transformer_model_ = Model(inputs=inputs,
                                        outputs=concatenated_features)
        self.locator_model_ = Model(inputs=inputs,
                                    outputs=concatenated_locations)
        self.model_.compile(loss=loss,
                            optimizer=self.optimizer,
                            metrics=metrics)

    def get_weights(self, layer_name=None):
        """Return model weights (or weights for a given layer if `layer_name`
        is provided).

        Parameters
        ----------
        layer_name: str or None (default: None)
            Name of the layer for which  weights should be returned.
            If None, all model weights are returned.
            Available layer names with weights are:

            - "shapelets_i_j" with i an integer for the shapelet id and j an
              integer for the dimension
            - "classification" for the final classification layer

        Returns
        -------
        list
            list of model (or layer) weights

        Examples
        --------
        >>> from tslearn.generators import random_walk_blobs
        >>> X, y = random_walk_blobs(n_ts_per_blob=100, sz=256, d=1, n_blobs=3)
        >>> clf = ShapeletModel(n_shapelets_per_size={10: 5}, max_iter=0,
        ...                     verbose=0)
        >>> clf.fit(X, y).get_weights("classification")[0].shape
        (5, 3)
        >>> clf.get_weights("shapelets_0_0")[0].shape
        (5, 10)
        >>> len(clf.get_weights("shapelets_0_0"))
        1
        """
        if layer_name is None:
            return self.model_.get_weights()
        else:
            return self.model_.get_layer(layer_name).get_weights()

    def set_weights(self, weights, layer_name=None):
        """Set model weights (or weights for a given layer if `layer_name`
        is provided).

        Parameters
        ----------
        weights: list of ndarrays
            Weights to set for the model / target layer

        layer_name: str or None (default: None)
            Name of the layer for which  weights should be set.
            If None, all model weights are set.
            Available layer names with weights are:

            - "shapelets_i_j" with i an integer for the shapelet id and j an
              integer for the dimension
            - "classification" for the final classification layer

        Examples
        --------
        >>> from tslearn.generators import random_walk_blobs
        >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=16, d=1, n_blobs=3)
        >>> clf = ShapeletModel(n_shapelets_per_size={3: 1}, max_iter=0,
        ...                     verbose=0)
        >>> _ = clf.fit(X, y)
        >>> weights_shapelet = [
        ...     numpy.array([[1, 2, 3]])
        ... ]
        >>> clf.set_weights(weights_shapelet, layer_name="shapelets_0_0")
        >>> clf.shapelets_as_time_series_
        array([[[1.],
                [2.],
                [3.]]])
        """
        if layer_name is None:
            return self.model_.set_weights(weights)
        else:
            return self.model_.get_layer(layer_name).set_weights(weights)

    def _get_tags(self):
        # This is added due to the fact that there are small rounding
        # errors in the `transform` method, while sklearn performs checks
        # that requires the output of transform to have less than 1e-9
        # difference between outputs of same input.
        return {'non_deterministic': True}


class SerializableShapeletModel(ShapeletModel):
    """Serializable variant of the Learning Time-Series Shapelets model.


    Learning Time-Series Shapelets was originally presented in [1]_.

    Parameters
    ----------
    n_shapelets_per_size: dict (default: None)
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value)
    max_iter: int (default: 10,000)
        Number of training epochs.

        .. versionchanged:: 0.3
            default value for max_iter is set to 10,000 instead of 100

    batch_size: int (default:256)
        Batch size to be used.
    verbose_level: {0, 1, 2} (default: 0)
        `keras` verbose level.

        .. deprecated:: 0.1
            min is deprecated in version 0.1 and will be
            removed in 0.2.

    verbose: {0, 1, 2} (default: 0)
        `keras` verbose level.
    learning_rate: float (default: 0.01)
        Learning rate to be used for the SGD optimizer.
    weight_regularizer: float or None (default: 0.)
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer. If 0, no regularization is performed.
    shapelet_length: float (default 0.15)
        The length of the shapelets, expressed as a fraction of the ts length
    total_lengths: int (default 3)
        The number of different shapelet lengths. Will extract shapelets of
        length i * shapelet_length for i in [1, total_lengths]
    random_state : int or None, optional (default: None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    shapelets_ : numpy.ndarray of objects, each object being a time series
        Set of time-series shapelets.

    shapelets_as_time_series_ : numpy.ndarray of shape (n_shapelets, sz_shp, \
            d) where `sz_shp` is the maximum of all  shapelet sizes
        Set of time-series shapelets formatted as a ``tslearn`` time series
        dataset.

    transformer_model_ : keras.Model
        Transforms an input dataset of timeseries into distances to the
        learned shapelets.

    locator_model_ : keras.Model
        Returns the indices where each of the shapelets can be found (minimal
        distance) within each of the timeseries of the input dataset.

    model_ : keras.Model
        Directly predicts the class probabilities for the input timeseries.

    Notes
    -----
        This implementation requires a dataset of equal-sized time series.

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=16, d=2, n_blobs=3)
    >>> clf = SerializableShapeletModel(n_shapelets_per_size={4: 5},
    ...                                 max_iter=1, verbose=0,
    ...                                 learning_rate=0.01)
    >>> _ = clf.fit(X, y)
    >>> clf.shapelets_.shape[0]
    5
    >>> clf.shapelets_[0].shape
    (4, 2)
    >>> clf.predict(X).shape
    (30,)
    >>> clf.predict_proba(X).shape
    (30, 3)
    >>> clf.transform(X).shape
    (30, 5)

    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    def __init__(self, n_shapelets_per_size=None,
                 max_iter=10000,
                 batch_size=256,
                 verbose=0,
                 verbose_level=None,
                 learning_rate=0.01,
                 weight_regularizer=0.,
                 shapelet_length=0.3,
                 total_lengths=3,
                 random_state=None):
        super(SerializableShapeletModel,
              self).__init__(n_shapelets_per_size=n_shapelets_per_size,
                             max_iter=max_iter,
                             batch_size=batch_size,
                             verbose=verbose,
                             verbose_level=verbose_level,
                             weight_regularizer=weight_regularizer,
                             shapelet_length=shapelet_length,
                             total_lengths=total_lengths,
                             random_state=random_state)
        self.learning_rate = learning_rate

    def _set_model_layers(self, X, ts_sz, d, n_classes):
        super(SerializableShapeletModel,
              self)._set_model_layers(X=X,
                                      ts_sz=ts_sz,
                                      d=d,
                                      n_classes=n_classes)
        K.set_value(self.model_.optimizer.lr, self.learning_rate)

    def set_params(self, **params):
        return super(SerializableShapeletModel, self).set_params(**params)
