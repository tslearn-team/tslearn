from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import (InputSpec, Dense, Conv1D, Layer, Input,
                                     concatenate, add)
from tensorflow.keras.metrics import (categorical_accuracy,
                                      categorical_crossentropy,
                                      binary_accuracy, binary_crossentropy)
from tensorflow.keras.utils import to_categorical
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Initializer
import tensorflow.keras.backend as K
import numpy
import tensorflow as tf

import warnings

from ..utils import to_time_series_dataset, check_dims, ts_size
from ..bases import BaseModelPackage, TimeSeriesBaseEstimator
from ..clustering import TimeSeriesKMeans
from ..preprocessing import TimeSeriesScalerMinMax

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class GlobalMinPooling1D(Layer):
    """Global min pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
        
    Examples
    --------
    >>> x = tf.constant([5.0, numpy.nan, 6.8, numpy.nan, numpy.inf])
    >>> x = tf.reshape(x, [1, 5, 1])
    >>> GlobalMinPooling1D()(x).numpy()
    array([[5.]], dtype=float32)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        inputs_without_nans = tf.where(tf.math.is_finite(inputs),
                                       inputs,
                                       tf.zeros_like(inputs) + numpy.inf)
        return tf.reduce_min(inputs_without_nans, axis=1)


class GlobalArgminPooling1D(Layer):
    """Global min pooling operation for temporal data.
    # Input shape
        3D tensor with shape: `(batch_size, steps, features)`.
    # Output shape
        2D tensor with shape:
        `(batch_size, features)`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

    def call(self, inputs, **kwargs):
        return K.cast(K.argmin(inputs, axis=1), dtype=K.floatx())


def _kmeans_init_shapelets(X, n_shapelets, shp_len, n_draw=10000):
    n_ts, sz, d = X.shape
    indices_ts = numpy.random.choice(n_ts, size=n_draw, replace=True)
    indices_time = numpy.array(
        [numpy.random.choice(ts_size(ts) - shp_len + 1, size=1)[0]
         for ts in X[indices_ts]]
    )
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
        return tf.convert_to_tensor(shapelets, dtype=K.floatx())

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
        if X is None:
            self.initializer = "uniform"
        else:
            self.initializer = KMeansShapeletInitializer(X)
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.n_shapelets, input_shape[2]),
                                      initializer=self.initializer,
                                      trainable=True)
        super().build(input_shape)

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
        base_config = super().get_config()
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
        Dictionary giving, for each shapelet length, the number of such
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


class LearningShapelets(ClassifierMixin, TransformerMixin,
                        BaseModelPackage, TimeSeriesBaseEstimator):
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
        the number of such shapelets to be trained (value). 
        If None, `grabocka_params_to_shapelet_size_dict` is used and the
        size used to compute is that of the shortest time series passed at fit 
        time.
        
    max_iter: int (default: 10,000)
        Number of training epochs.

        .. versionchanged:: 0.3
            default value for max_iter is set to 10,000 instead of 100

    batch_size: int (default: 256)
        Batch size to be used.

    verbose: {0, 1, 2} (default: 0)
        `keras` verbose level.
    
    optimizer: str or keras.optimizers.Optimizer (default: "sgd")
        `keras` optimizer to use for training.
    
    weight_regularizer: float or None (default: 0.)
        Strength of the L2 regularizer to use for training the classification
        (softmax) layer. If 0, no regularization is performed.
    
    shapelet_length: float (default: 0.15)
        The length of the shapelets, expressed as a fraction of the time 
        series length.
        Used only if `n_shapelets_per_size` is None.
    
    total_lengths: int (default: 3)
        The number of different shapelet lengths. Will extract shapelets of
        length i * shapelet_length for i in [1, total_lengths]
        Used only if `n_shapelets_per_size` is None.
        
    max_size: int or None (default: None)
        Maximum size for time series to be fed to the model. If None, it is 
        set to the size (number of timestamps) of the training time series.
        
    scale: bool (default: False)
        Whether input data should be scaled for each feature of each time 
        series to lie in the [0-1] interval.
        Default for this parameter is set to `False` in version 0.4 to ensure
        backward compatibility, but is likely to change in a future version.        
    
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

    history_ : dict
        Dictionary of losses and metrics recorded during fit.

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=16, d=2, n_blobs=3)
    >>> clf = LearningShapelets(n_shapelets_per_size={4: 5}, 
    ...                         max_iter=1, verbose=0)
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
                 optimizer="sgd",
                 weight_regularizer=0.,
                 shapelet_length=0.15,
                 total_lengths=3,
                 max_size=None,
                 scale=False,
                 random_state=None):
        self.n_shapelets_per_size = n_shapelets_per_size
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.optimizer = optimizer
        self.weight_regularizer = weight_regularizer
        self.shapelet_length = shapelet_length
        self.total_lengths = total_lengths
        self.max_size = max_size
        self.scale = scale
        self.random_state = random_state

        if not scale:
            warnings.warn("The default value for 'scale' is set to False "
                          "in version 0.4 to ensure backward compatibility, "
                          "but is likely to change in a future version.",
                          FutureWarning)

    @property
    def _n_shapelet_sizes(self):
        return len(self.n_shapelets_per_size_)

    @property
    def shapelets_(self):
        if self.n_shapelets_per_size is None:
            n_shapelets_per_size = self.n_shapelets_per_size_
        else:
            n_shapelets_per_size = self.n_shapelets_per_size
        total_n_shp = sum(n_shapelets_per_size.values())
        shapelets = numpy.empty((total_n_shp, ), dtype=object)
        idx = 0
        for i, shp_sz in enumerate(sorted(n_shapelets_per_size.keys())):
            n_shp = n_shapelets_per_size[shp_sz]
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
        >>> model = LearningShapelets(n_shapelets_per_size={3: 2, 4: 1},
        ...                       max_iter=1)
        >>> _ = model.fit(X, y)
        >>> model.shapelets_as_time_series_.shape
        (3, 4, 1)
        """
        if self.n_shapelets_per_size is None:
            n_shapelets_per_size = self.n_shapelets_per_size_
        else:
            n_shapelets_per_size = self.n_shapelets_per_size
        total_n_shp = sum(n_shapelets_per_size.values())
        shp_sz = max(n_shapelets_per_size.keys())
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
        X, y = check_X_y(X, y, allow_nd=True, force_all_finite=False)
        X = self._preprocess_series(X)
        X = check_dims(X)
        self._check_series_length(X)

        if self.random_state is not None:
            tf.keras.utils.set_random_seed(seed=self.random_state)

        n_ts, sz, d = X.shape
        self._X_fit_dims = X.shape

        self.model_ = None
        self.transformer_model_ = None
        self.locator_model_ = None
        self.d_ = d

        y_ = self._preprocess_labels(y)
        n_labels = len(self.classes_)

        if self.n_shapelets_per_size is None:
            sizes = grabocka_params_to_shapelet_size_dict(n_ts,
                                                          self._min_sz_fit,
                                                          n_labels,
                                                          self.shapelet_length,
                                                          self.total_lengths)
            self.n_shapelets_per_size_ = sizes
        else:
            self.n_shapelets_per_size_ = self.n_shapelets_per_size

        self._set_model_layers(X=X, ts_sz=sz, d=d, n_classes=n_labels)
        self._set_weights_false_conv(d=d)
        h = self.model_.fit(
            [X[:, :, di].reshape((n_ts, sz, 1)) for di in range(d)], y_,
            batch_size=self.batch_size, epochs=self.max_iter,
            verbose=self.verbose
        )
        self.history_ = h.history
        self.n_iter_ = len(self.history_.get("loss", []))
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
        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = self._preprocess_series(X)
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        self._check_series_length(X)

        y_ind = self.predict_proba(X).argmax(axis=1)
        y_label = numpy.array(
            [self.classes_[ind] for ind in y_ind]
        )
        return y_label

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
        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = self._preprocess_series(X)
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        self._check_series_length(X)

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
        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = self._preprocess_series(X)
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        self._check_series_length(X)

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
        >>> clf = LearningShapelets(n_shapelets_per_size={3: 1}, max_iter=0,
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
        check_is_fitted(self, '_X_fit_dims')
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = self._preprocess_series(X)
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        self._check_series_length(X)

        n_ts, sz, d = X.shape
        locations = self.locator_model_.predict(
            [X[:, :, di].reshape((n_ts, sz, 1)) for di in range(self.d_)],
            batch_size=self.batch_size, verbose=self.verbose
        )
        return locations.astype(int)

    def _check_series_length(self, X):
        """Ensures that time series in X matches the following requirements:
        
        - their length is greater than the size of the longest shapelet
        - (at predict time) their length is lower than the maximum allowed 
        length, as set by self.max_size
        """
        sizes = numpy.array([ts_size(Xi) for Xi in X])
        self._min_sz_fit = sizes.min()

        if self.n_shapelets_per_size is not None:
            max_sz_shp = max(self.n_shapelets_per_size.keys())
            if max_sz_shp > self._min_sz_fit:
                raise ValueError("Sizes in X do not match maximum "
                                 "shapelet size: there is at least one "
                                 "series in X that is shorter than one of the "
                                 "shapelets. Shortest time series is of "
                                 "length {} and longest shapelet is of length "
                                 "{}".format(self._min_sz_fit, max_sz_shp))

        if hasattr(self, 'model_') or self.max_size is not None:
            # Model is already fitted
            max_sz_X = sizes.max()

            if hasattr(self, 'model_'):
                max_size = self._X_fit_dims[1]
            else:
                max_size = self.max_size
            if max_size < max_sz_X:
                raise ValueError("Sizes in X do not match maximum allowed "
                                 "size as set by max_size. "
                                 "Longest time series is of "
                                 "length {} and max_size is "
                                 "{}".format(max_sz_X, max_size))

    def _preprocess_series(self, X):
        if self.scale:
            X = TimeSeriesScalerMinMax().fit_transform(X)
        else:
            X = to_time_series_dataset(X)
        if self.max_size is not None and self.max_size != X.shape[1]:
            if X.shape[1] > self.max_size:
                raise ValueError(
                    "Cannot feed model with series of length {} "
                    "max_size is {}".format(X.shape[1], self.max_size)
                )
            X_ = numpy.zeros((X.shape[0], self.max_size, X.shape[2]))
            X_[:, :X.shape[1]] = X
            X_[:, X.shape[1]:] = numpy.nan
            return X_
        else:
            return X

    def _preprocess_labels(self, y):
        self.classes_ = unique_labels(y)
        n_labels = len(self.classes_)
        if n_labels == 1:
            raise ValueError("Classifier can't train when only one class "
                             "is present.")
        if self.classes_.dtype in [numpy.int32, numpy.int64]:
            self.label_to_ind_ = {int(lab): ind
                                  for ind, lab in enumerate(self.classes_)}
        else:
            self.label_to_ind_ = {lab: ind
                                  for ind, lab in enumerate(self.classes_)}
        y_ind = numpy.array(
            [self.label_to_ind_[lab] for lab in y]
        )
        y_ = to_categorical(y_ind)
        if n_labels == 2:
            y_ = y_[:, 1:]  # Keep only indicator of positive class
        return y_

    def _build_auxiliary_models(self):
        check_is_fitted(self, 'model_')

        inputs = self.model_.inputs
        concatenated_features = self.model_.get_layer("classification").input

        self.transformer_model_ = Model(inputs=inputs,
                                        outputs=concatenated_features)
        self.transformer_model_.compile(loss="mean_squared_error",
                                        optimizer=self.optimizer)

        min_pool_inputs = [self.model_.get_layer("min_pooling_%d" % i).input
                           for i in range(self._n_shapelet_sizes)]

        pool_layers_locations = [
            GlobalArgminPooling1D(name="min_pooling_%d" % i)(pool_input)
            for i, pool_input in enumerate(min_pool_inputs)
        ]
        if self._n_shapelet_sizes > 1:
            concatenated_locations = concatenate(pool_layers_locations)
        else:
            concatenated_locations = pool_layers_locations[0]

        self.locator_model_ = Model(inputs=inputs,
                                    outputs=concatenated_locations)
        self.locator_model_.compile(loss="mean_squared_error",
                                    optimizer=self.optimizer)

    def _set_weights_false_conv(self, d):
        shapelet_sizes = sorted(self.n_shapelets_per_size_.keys())
        for i, sz in enumerate(shapelet_sizes):
            for di in range(d):
                layer = self.model_.get_layer("false_conv_%d_%d" % (i, di))
                layer.set_weights([numpy.eye(sz).reshape((sz, 1, sz))])

    def _set_model_layers(self, X, ts_sz, d, n_classes):
        inputs = [Input(shape=(self.max_size, 1),
                        name="input_%d" % di)
                  for di in range(d)]
        shapelet_sizes = sorted(self.n_shapelets_per_size_.keys())
        pool_layers = []
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
            pool_layers.append(gp)
        if len(shapelet_sizes) > 1:
            concatenated_features = concatenate(pool_layers)
        else:
            concatenated_features = pool_layers[0]

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
        self.model_.compile(loss=loss,
                            optimizer=self.optimizer,
                            metrics=metrics)
        self._build_auxiliary_models()

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
        >>> clf = LearningShapelets(n_shapelets_per_size={10: 5}, max_iter=0,
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
        >>> clf = LearningShapelets(n_shapelets_per_size={3: 1}, max_iter=0,
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

    def _is_fitted(self):
        check_is_fitted(self, 'model_')
        return True

    def _get_model_params(self):
        """Get model parameters that are sufficient to recapitulate it."""
        params = super()._get_model_params()
        params.update({"_X_fit_dims": self._X_fit_dims,
                       "model_": self.model_.to_json(),
                       "model_weights_": self.get_weights()})
        return params

    @staticmethod
    def _organize_model(cls, model):
        """
        Instantiate the model with all hyper-parameters,
        set all model parameters and then return the model.
        Do not use directly. Use the designated classmethod to load a model.
        Parameters
        ----------
        cls : instance of model that inherits from `BaseModelPackage`
            a model instance
        model : dict
            Model dict containing hyper-parameters and model-parameters
        Returns
        -------
        model: instance of model that inherits from `BaseModelPackage`
            instance of the model class with hyper-parameters and
            model parameters set from the passed model dict
        """

        model_params = model.pop('model_params')
        hyper_params = model.pop('hyper_params')  # hyper-params

        # instantiate with hyper-parameters
        inst = cls(**hyper_params)

        if "model_" in model_params.keys():
            # set all model params
            inst.model_ = model_from_json(
                model_params.pop("model_"),
                custom_objects={
                    "LocalSquaredDistanceLayer": LocalSquaredDistanceLayer,
                    "GlobalMinPooling1D": GlobalMinPooling1D
                }
            )
            inst.set_weights(model_params.pop("model_weights_"))
        for p in model_params.keys():
            setattr(inst, p, model_params[p])
        inst._X_fit_dims = tuple(inst._X_fit_dims)
        inst._build_auxiliary_models()

        return inst

    def _more_tags(self):
        # This is added due to the fact that there are small rounding
        # errors in the `transform` method, while sklearn performs checks
        # that requires the output of transform to have less than 1e-9
        # difference between outputs of same input.
        return {'allow_nan': True, 'allow_variable_length': True}


ShapeletModel = LearningShapelets
