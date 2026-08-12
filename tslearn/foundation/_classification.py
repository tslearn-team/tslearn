"""Re-use of pre-trained time series models for classification."""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesMixin
from tslearn.utils import check_array, to_time_series_dataset

from ._embedding import (
    LAYOUTS,
    TimeSeriesFoundationEmbedder,
    _check_probe_pooling,
    _require_torch,
)


class LinearProbeClassifier(TimeSeriesMixin, ClassifierMixin, BaseEstimator):
    """Classify time series with a linear head on a frozen pre-trained model.

    The pre-trained model is used as a frozen feature extractor and a linear
    classifier is fitted on the resulting representations. This is the standard
    protocol used to assess how much class information a pre-trained model has
    learnt [1]_: because the head is linear and the backbone is frozen, the
    accuracy it reaches measures how linearly separable the classes already are
    in the representation space.

    It is also a practical classifier in its own right, as it needs no
    backpropagation through the pre-trained model and therefore trains in
    seconds even on models counting hundreds of millions of parameters.

    Parameters
    ----------
        model : torch.nn.Module
          A pre-trained model used as a frozen feature extractor.
        probe : sklearn estimator or None (default: None)
          The head fitted on top of the frozen representations. When None, a
          :class:`sklearn.linear_model.LogisticRegression` is used. Any
          scikit-learn classifier can be passed instead; note that using a
          non-linear one makes the resulting accuracy an estimate of predictive
          performance rather than of linear separability. Frozen
          representations sometimes have a very small variance, in which case
          the default regularization is too strong; passing
          ``make_pipeline(StandardScaler(), LogisticRegression())`` is then
          usually enough to fix it.
        context_length : int or None (default: None)
          When set, only the last ``context_length`` timestamps of each series
          are fed to the model.
        layer : int or None (default: None)
          Layer to probe. When None, the model's output hidden state is used.
          When an integer, a forward hook is placed on the corresponding block
          of the model's layer stack. Intermediate layers often carry more
          class information than the last ones, which tend to specialize
          towards the pre-training objective, so this is worth tuning.
        layers_path : str or None (default: None)
          Dotted path to the stack of layers, e.g. ``"encoder.block"``. When
          None, the stack is auto-detected.
        pooling : {"mean", "max", "cls", "last", "flatten"} (default: "mean")
          How token representations are aggregated into one vector per series.
          A probe needs a flat feature matrix, so ``pooling=None`` is not
          accepted here.
        tokens : slice, (start, stop) pair or None (default: None)
          Which tokens to keep before pooling, see
          :class:`~tslearn.foundation.TimeSeriesFoundationEmbedder`. Restricting
          the average to the tokens that actually represent the series, e.g.
          ``tokens=(0, -2)`` for Chronos-2, avoids diluting it with class or
          forecast tokens.
        cls_index : int (default: 0)
          Index of the token selected when ``pooling="cls"``.
        input_layout : {"univariate", "channels_last", "channels_first"} (default: "univariate")
          How multivariate series are handed over to the model.
        input_name : str or None (default: None)
          Name of the ``forward`` argument receiving the context values.
        model_kwargs : dict or None (default: None)
          Extra keyword arguments passed to every ``forward`` call.
        batch_size : int (default: 32)
          Number of series embedded at once.
        device : str or None (default: None)
          Device on which inference is run.
        verbose : int (default: 0)
          When positive, prints progress information.

    Attributes
    ----------
        embedder_ : TimeSeriesFoundationEmbedder
          The frozen feature extractor.
        probe_ : sklearn estimator
          The fitted linear head.
        classes_ : array of shape=(n_classes,)
          Class labels known to the classifier.
        n_features_in_ : int
          Number of features (channels) of the series seen during fit.

    See Also
    --------
        LinearProbeForecaster: Linear probing for time series forecasting.
        TimeSeriesFoundationEmbedder: The underlying feature extractor.

    Examples
    --------
    >>> model = LinearProbeClassifier(backbone, layer=-2)  # doctest: +SKIP
    >>> model.fit(X_train, y_train).score(X_test, y_test)  # doctest: +SKIP
    0.93

    References
    ----------
    .. [1] G. Alain and Y. Bengio. Understanding intermediate layers using
      linear classifier probes. ICLR Workshop, 2017.

    """

    def __init__(
        self,
        model,
        probe=None,
        context_length=None,
        layer=None,
        layers_path=None,
        pooling="mean",
        tokens=None,
        cls_index=0,
        input_layout="univariate",
        input_name=None,
        model_kwargs=None,
        batch_size=32,
        device=None,
        verbose=0,
    ):
        self.model = model
        self.probe = probe
        self.context_length = context_length
        self.layer = layer
        self.layers_path = layers_path
        self.pooling = pooling
        self.tokens = tokens
        self.cls_index = cls_index
        self.input_layout = input_layout
        self.input_name = input_name
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def _make_embedder(self):
        return TimeSeriesFoundationEmbedder(
            model=self.model,
            layer=self.layer,
            layers_path=self.layers_path,
            pooling=self.pooling,
            tokens=self.tokens,
            cls_index=self.cls_index,
            input_layout=self.input_layout,
            context_length=self.context_length,
            input_name=self.input_name,
            model_kwargs=self.model_kwargs,
            batch_size=self.batch_size,
            device=self.device,
            verbose=self.verbose,
        )

    def _make_probe(self):
        if self.probe is None:
            return LogisticRegression(max_iter=1000)
        return clone(self.probe)

    def _check_input(self, X):
        if self.input_layout not in LAYOUTS:
            raise ValueError(
                f"`input_layout` must be one of {LAYOUTS}, got "
                f"'{self.input_layout}'."
            )
        _check_probe_pooling(self.pooling)
        X = check_array(X, allow_nd=True, force_all_finite=True)
        return to_time_series_dataset(X)

    def fit(self, X, y):
        """Fit a linear classifier on top of the frozen pre-trained model.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.
            y : array-like of shape=(n_ts,)
                Class labels.

        Returns
        -------
            self
                The fitted estimator

        """
        _require_torch()
        X = self._check_input(X)
        y = np.asarray(y)
        check_classification_targets(y)
        if len(y) != X.shape[0]:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {X.shape[0]} "
                f"and {len(y)}."
            )

        self.embedder_ = self._make_embedder()
        embeddings = self.embedder_.fit_transform(X)

        self.probe_ = self._make_probe()
        self.probe_.fit(embeddings, y)

        self.classes_ = np.asarray(self.probe_.classes_)
        self.n_features_in_ = X.shape[2]
        return self

    def _transform(self, X):
        check_is_fitted(self, "probe_")
        X = self._check_input(X)
        if X.shape[2] != self.n_features_in_:
            raise ValueError(
                f"Series with {self.n_features_in_} features were expected, got "
                f"{X.shape[2]}."
            )
        return self.embedder_.transform(X)

    def transform(self, X):
        """Return the frozen representations the classifier operates on.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.

        Returns
        -------
            array of shape=(n_ts, embedding_size)
              Frozen representations of the input series.

        """
        return self._transform(X)

    def predict(self, X):
        """Predict the class of each time series.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.

        Returns
        -------
            array of shape=(n_ts,)
              Predicted class labels.

        """
        return self.probe_.predict(self._transform(X))

    def predict_proba(self, X):
        """Predict class probabilities for each time series.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.

        Returns
        -------
            array of shape=(n_ts, n_classes)
              Predicted class probabilities, ordered as ``classes_``.

        """
        check_is_fitted(self, "probe_")
        if not hasattr(self.probe_, "predict_proba"):
            raise AttributeError(
                f"The probe {type(self.probe_).__name__} does not expose "
                "`predict_proba`."
            )
        return self.probe_.predict_proba(self._transform(X))

    def decision_function(self, X):
        """Return the decision function of the linear head.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.

        Returns
        -------
            array of shape=(n_ts,) or (n_ts, n_classes)
              Confidence scores.

        """
        check_is_fitted(self, "probe_")
        if not hasattr(self.probe_, "decision_function"):
            raise AttributeError(
                f"The probe {type(self.probe_).__name__} does not expose "
                "`decision_function`."
            )
        return self.probe_.decision_function(self._transform(X))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags
