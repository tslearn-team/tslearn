"""
The :mod:`tslearn.shapelets` module gathers Shapelet-based algorithms.

It depends on the `keras` library for optimization.
"""

from keras.models import Model
from keras.layers import Dense, Conv1D, Layer, Input, concatenate
from keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.utils import to_categorical
from keras.regularizers import l2
import keras.backend as K
from keras.engine import InputSpec
import numpy

from tslearn.utils import to_time_series, to_time_series_dataset

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


class LocalSquaredDistanceLayer(Layer):
    """Pairwise (squared) distance computation between local patches and shapelets
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        3D tensor with shape:
        `(batch_size, steps, n_shapelets)`
    """
    def __init__(self, n_shapelets, **kwargs):
        self.n_shapelets = n_shapelets
        super(LocalSquaredDistanceLayer, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.n_shapelets, input_shape[2]),
                                      initializer='uniform',
                                      trainable=True)
        super(LocalSquaredDistanceLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        # (x - y)^2 = x^2 + y^2 - 2 * x * y
        x_sq = K.expand_dims(K.sum(x ** 2, axis=2), axis=-1)
        y_sq = K.reshape(K.sum(self.kernel ** 2, axis=1), (1, 1, self.n_shapelets))
        xy = K.dot(x, K.transpose(self.kernel))
        return x_sq + y_sq - 2 * xy

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.n_shapelets


def grabocka_params_to_shapelet_size_dict(ts_sz, n_classes, l, r):
    """Compute number and length of shapelets the way it is done in [1]_.

    Parameters
    ----------
    ts_sz: int
        Length of time series in the dataset
    n_classes: int
        Number of classes in the dataset
    l: float
        Fraction of the length of time series to be used for base shapelet length
    r: int
        Number of different shapelet lengths to use

    Returns
    -------
    dict
        Dictionnary giving, for each shapelet length, the number of such shapelets to be generated

    Examples
    --------
    >>> d = grabocka_params_to_shapelet_size_dict(ts_sz=100, n_classes=3, l=0.1, r=2)
    >>> keys = sorted(d.keys())
    >>> print(keys)
    [10, 20]
    >>> print([d[k] for k in keys])
    [3, 3]


    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    base_size = int(l * ts_sz)
    d = {}
    for sz_idx in range(r):
        shp_sz = base_size * (sz_idx + 1)
        n_shapelets = int(numpy.log10(ts_sz - shp_sz + 1) * (n_classes - 1))
        d[shp_sz] = n_shapelets
    return d


class ShapeletModel:
    """Learning Time-Series Shapelets model as presented in [1]_.

    This implementation only accepts mono-dimensional time series as inputs.

    Parameters
    ----------
    n_shapelets_per_size: dict
        Dictionary giving, for each shapelet size (key),
        the number of such shapelets to be trained (value)
    max_iter: int (default: 1000)
        Number of training epochs.
    batch_size: int (default:256)
        Batch size to be used.
    verbose_level: {0, 1, 2} (default: 2)
        `keras` verbose level.
    optimizer: str or keras.optimizers.Optimizer (default: "sgd")
        `keras` optimizer to use for training.
    weight_regularizer: float or None (default: None)
        `keras` regularizer to use for training the classification (softmax) layer.
        If None, no regularization is performed.

    Attributes
    ----------
    shapelets: numpy.ndarray
        Set of time-series shapelets

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=100, sz=256, d=1, n_blobs=3)
    >>> clf = ShapeletModel(n_shapelets_per_size={10: 5}, max_iter=1, verbose_level=0)
    >>> clf.fit(X, y).shapelets_.shape
    (5,)
    >>> clf.shapelets_[0].shape
    (10, 1)
    >>> clf.predict(X).shape
    (300,)
    >>> clf.transform(X).shape
    (300, 5, 1)
    >>> clf2 = ShapeletModel(n_shapelets_per_size={10: 5, 20: 10}, max_iter=1, verbose_level=0)
    >>> clf2.fit(X, y).shapelets_.shape
    (15,)
    >>> clf2.shapelets_[0].shape
    (10, 1)
    >>> clf2.shapelets_[5].shape
    (20, 1)
    >>> clf2.predict(X).shape
    (300,)
    >>> clf2.transform(X).shape
    (300, 15, 1)


    References
    ----------
    .. [1] J. Grabocka et al. Learning Time-Series Shapelets. SIGKDD 2014.
    """
    def __init__(self, n_shapelets_per_size,
                 max_iter=1000,
                 batch_size=256,
                 verbose_level=2,
                 optimizer="sgd",
                 weight_regularizer=0.):
        self.n_shapelets_per_size = n_shapelets_per_size
        self.n_classes = None
        self.optimizer = optimizer
        self.epochs = max_iter
        self.weight_regularizer = weight_regularizer
        self.model = None
        self.transformer_model = None
        self.batch_size = batch_size
        self.verbose_level = verbose_level
        self.categorical_y = False

    @property
    def _n_shapelet_sizes(self):
        return len(self.n_shapelets_per_size)

    @property
    def shapelets_(self):
        total_n_shp = sum(self.n_shapelets_per_size.values())
        shapelets = numpy.empty((total_n_shp, ), dtype=object)
        idx = 0
        for i in range(self._n_shapelet_sizes):
            for shp in self.model.get_layer("shapelets_%d" % i).get_weights()[0]:
                shapelets[idx] = to_time_series(shp)
                idx += 1
        assert idx == total_n_shp
        return shapelets

    def fit(self, X, y):
        n_ts, sz, d = X.shape
        assert(d == 1)
        if y.ndim == 1:
            y_ = to_categorical(y)
        else:
            y_ = y
            self.categorical_y = True
        n_classes = y_.shape[1]
        self._set_model_layers(ts_sz=sz, d=d, n_classes=n_classes)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=self.optimizer,
                           metrics=[categorical_accuracy,
                                    categorical_crossentropy])
        self.transformer_model.compile(loss="mean_squared_error",
                                       optimizer=self.optimizer)
        self._set_weights_false_conv(d=d)
        self.model.fit(X, y_,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=self.verbose_level)
        return self

    def predict(self, X):
        categorical_preds = self.model.predict(X,
                                               batch_size=self.batch_size,
                                               verbose=self.verbose_level)
        if self.categorical_y:
            return categorical_preds
        else:
            return categorical_preds.argmax(axis=1)

    def transform(self, X):
        pred = self.transformer_model.predict(X,
                                              batch_size=self.batch_size,
                                              verbose=self.verbose_level)
        return to_time_series_dataset(pred)

    def _set_weights_false_conv(self, d):
        shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        for i, sz in enumerate(sorted(shapelet_sizes)):
            weights_false_conv = numpy.empty((sz, d, sz))
            for di in range(d):
                weights_false_conv[:, di, :] = numpy.eye(sz)
            layer = self.model.get_layer("false_conv_%d" % i)
            layer.set_weights([weights_false_conv])

    def _set_model_layers(self, ts_sz, d, n_classes):
        inputs = Input(shape=(ts_sz, d), name="input")
        shapelet_sizes = sorted(self.n_shapelets_per_size.keys())
        pool_layers = []
        for i, sz in enumerate(sorted(shapelet_sizes)):
            transformer_layer = Conv1D(filters=sz,
                                       kernel_size=sz,
                                       trainable=False,
                                       use_bias=False,
                                       name="false_conv_%d" % i)(inputs)
            shapelet_layer = LocalSquaredDistanceLayer(self.n_shapelets_per_size[sz],
                                                       name="shapelets_%d" % i)(transformer_layer)
            pool_layers.append(GlobalMinPooling1D(name="min_pooling_%d" % i)(shapelet_layer))
        if len(shapelet_sizes) > 1:
            concatenated_features = concatenate(pool_layers)
        else:
            concatenated_features = pool_layers[0]
        if self.weight_regularizer > 0.:
            outputs = Dense(units=n_classes,
                            activation="softmax",
                            kernel_regularizer=l2(self.weight_regularizer),
                            name="softmax")(concatenated_features)
        else:
            outputs = Dense(units=n_classes,
                            activation="softmax",
                            name="softmax")(concatenated_features)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.transformer_model = Model(inputs=inputs, outputs=concatenated_features)

    def get_weights(self, layer_name=None):
        """Return model weights (or weights for a given layer if `layer_name` is provided).

        Parameters
        ----------
        layer_name: str or None (default: None)
            Name of the layer for which  weights should be returned.
            If None, all model weights are returned.
            Available layer names with weights are:
            - "shapelets_i" with i an integer for the sets of shapelets
              corresponding to each shapelet size (sorted in ascending order)
            - "softmax" for the final classification layer

        Returns
        -------
        list
            list of model (or layer) weights

        Examples
        --------
        >>> from tslearn.generators import random_walk_blobs
        >>> X, y = random_walk_blobs(n_ts_per_blob=100, sz=256, d=1, n_blobs=3)
        >>> clf = ShapeletModel(n_shapelets_per_size={10: 5}, max_iter=1, verbose_level=0)
        >>> clf.fit(X, y).get_weights("softmax")[0].shape
        (5, 3)
        """
        if layer_name is None:
            return self.model.get_weights()
        else:
            return self.model.get_layer(layer_name).get_weights()
