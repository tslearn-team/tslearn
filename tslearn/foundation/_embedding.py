"""Model-agnostic extraction of representations from pre-trained time series models."""

import inspect
import warnings

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from tslearn.bases import TimeSeriesMixin
from tslearn.utils import check_array, to_time_series_dataset

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


#: Names of ``forward`` arguments that are commonly used by pre-trained time
#: series models to receive the raw context values.
CANDIDATE_INPUT_NAMES = (
    "context",
    "past_values",
    "past_target",
    "input_values",
    "inputs",
    "x_enc",
    "series",
    "x",
)

#: Attribute names under which model outputs commonly expose hidden states.
CANDIDATE_HIDDEN_STATE_NAMES = (
    "last_hidden_state",
    "hidden_states",
    "encoder_last_hidden_state",
)

#: Supported ways of aggregating token representations. ``None``, for which
#: ``"none"`` is accepted as an alias, keeps them all and turns the estimator
#: into a time series to time series transform.
POOLINGS = ("mean", "max", "cls", "last", "flatten", "none", None)

#: Supported layouts for the array handed over to the wrapped model.
LAYOUTS = ("univariate", "channels_last", "channels_first")


def _normalize_pooling(pooling):
    return None if pooling == "none" else pooling


def _check_probe_pooling(pooling):
    """Reject the poolings that do not yield a flat feature matrix."""
    if _normalize_pooling(pooling) is None:
        raise ValueError(
            "A linear probe needs one feature vector per series, so "
            "`pooling=None` is not supported here. Use another pooling, or "
            "TimeSeriesFoundationEmbedder directly if you want to keep one "
            "representation per token."
        )


def _token_slice(tokens):
    """Turn the ``tokens`` parameter into a slice over the token axis."""
    if tokens is None:
        return slice(None)
    if isinstance(tokens, slice):
        return tokens
    if isinstance(tokens, (tuple, list)) and len(tokens) == 2:
        start, stop = tokens
        if all(bound is None or isinstance(bound, (int, np.integer))
               for bound in (start, stop)):
            return slice(start, stop)
    raise ValueError(
        "`tokens` must be None, a slice, or a (start, stop) pair of integers, "
        f"got {tokens!r}."
    )


def _layout_to_model_input(X, layout):
    """Lay a ``(n_ts, sz, d)`` dataset out as expected by the wrapped model.

    With the ``"univariate"`` layout, channels are unfolded along the batch
    axis so that the returned array has shape ``(n_ts * d, sz)``, series
    ``i`` and channel ``k`` sitting at row ``i * d + k``.
    """
    n_ts, sz, d = X.shape
    if layout == "univariate":
        return np.ascontiguousarray(np.swapaxes(X, 1, 2)).reshape(n_ts * d, sz)
    if layout == "channels_first":
        return np.ascontiguousarray(np.swapaxes(X, 1, 2))
    if layout == "channels_last":
        return np.ascontiguousarray(X)
    raise ValueError(f"`input_layout` must be one of {LAYOUTS}, got '{layout}'.")


def _require_torch():
    if torch is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required by the tslearn.foundation module. "
            "Install it with `pip install torch`."
        )


def _resolve_attribute_path(model, path):
    """Return the sub-module of ``model`` designated by a dotted ``path``."""
    module = model
    for attribute in path.split("."):
        if attribute.isdigit():
            module = module[int(attribute)]
        else:
            if not hasattr(module, attribute):
                raise AttributeError(
                    f"Could not resolve `layers_path='{path}'`: the model has no "
                    f"attribute '{attribute}'."
                )
            module = getattr(module, attribute)
    return module


def _autodetect_layers(model):
    """Find the container of repeated blocks of a transformer-like model.

    The heuristic looks for the longest :class:`torch.nn.ModuleList` whose
    children all share the same type, which is how stacks of identical
    encoder/decoder blocks are declared in essentially every implementation
    published on the Hugging Face Hub.
    """
    _require_torch()
    best_path, best_modules = None, None
    for path, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) < 2:
            continue
        types = {type(block) for block in module}
        if len(types) > 1:
            continue
        if best_modules is None or len(module) > len(best_modules):
            best_path, best_modules = path, module
    if best_modules is None:
        raise ValueError(
            "Could not automatically locate the stack of layers of the provided "
            "model. Pass an explicit `layers_path` (e.g. 'encoder.block') to "
            "select the layers to probe."
        )
    return best_path, best_modules


def _as_tensor(output):
    """Extract the hidden state tensor out of an arbitrary module output."""
    _require_torch()
    if isinstance(output, torch.Tensor):
        return output
    for name in CANDIDATE_HIDDEN_STATE_NAMES:
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)) and len(value) > 0:
            return value[-1]
    if isinstance(output, dict):
        for name in CANDIDATE_HIDDEN_STATE_NAMES:
            value = output.get(name)
            if isinstance(value, torch.Tensor):
                return value
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 3:
                return item
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(
        f"Could not extract a hidden state tensor out of an output of type "
        f"{type(output).__name__}. Models that only return their task-specific "
        "outputs, such as forecasts, expose no hidden state to pool: select an "
        "intermediate layer with the `layer` parameter, so that "
        "representations are read through a forward hook instead."
    )


class TimeSeriesFoundationEmbedder(TimeSeriesMixin, TransformerMixin, BaseEstimator):
    """Turn a frozen pre-trained time series model into a feature extractor.

    This transformer runs a (typically pre-trained) PyTorch model in inference
    mode and pools its hidden states into a single vector per time series. It
    is the building block used by
    :class:`~tslearn.foundation.LinearProbeForecaster` and
    :class:`~tslearn.foundation.LinearProbeClassifier`, and can also be used on
    its own, for instance inside a :class:`sklearn.pipeline.Pipeline`.

    The transformer is deliberately agnostic to any particular model
    implementation. It only assumes that:

    * ``model`` is a :class:`torch.nn.Module`;
    * its ``forward`` method accepts the raw context values through one
      argument (auto-detected among common names, see ``input_name``);
    * it returns, or internally computes, hidden states of shape
      ``(batch, n_tokens, dim)``.

    Parameters
    ----------
        model : torch.nn.Module
          A pre-trained model used as a frozen feature extractor. Its
          parameters are never updated by this class.
        layer : int or None (default: None)
          Which layer to read representations from. When None, the output of
          the model itself is used, which is usually the last hidden state
          after the final normalization layer. When an integer is given, a
          forward hook is placed on the corresponding block of the model's
          layer stack, so that ``layer=0`` probes the first block and
          ``layer=-1`` the last one. Probing intermediate layers is often
          preferable, as the last layers of a pre-trained model tend to
          specialize towards its pre-training objective [1]_.
        layers_path : str or None (default: None)
          Dotted path to the module holding the stack of layers, e.g.
          ``"encoder.block"``. Only used when ``layer`` is an integer. When
          None, the stack is auto-detected as the longest list of identical
          sub-modules.
        pooling : {"mean", "max", "cls", "last", "flatten", None} (default: "mean")
          How to aggregate the ``n_tokens`` representations of a series into a
          single vector. ``"cls"`` selects the single token at index
          ``cls_index``, which is how models exposing a class (or register)
          token are usually probed. ``"last"`` selects the last token and
          ``"flatten"`` concatenates them all, which yields a much larger, and
          context-length dependent, feature vector.

          ``None``, for which the string ``"none"`` is accepted as an alias,
          applies no pooling at all: :meth:`transform` then returns a time
          series dataset of shape ``(n_ts, n_tokens, d * dim)`` rather than a
          flat feature matrix, which turns this estimator into a time series to
          time series transform. Combined with ``tokens``, this is a way to
          obtain one representation per timestep, or per patch of timesteps,
          that can be fed to any other tslearn estimator. Note that the
          resulting series are not accepted by the linear probes of this
          module, which need a flat feature matrix.
        tokens : slice, (start, stop) pair or None (default: None)
          Which tokens to keep before pooling. Models often emit tokens that do
          not represent the input series, such as class, register or forecast
          tokens; excluding them is usually what one wants, both to build clean
          per-timestep representations and to avoid polluting an average.
          Chronos-2, for instance, emits its context tokens first, so
          ``tokens=(0, -2)`` drops its trailing register and forecast tokens.
          When None, all tokens are kept. This parameter does not affect
          ``pooling="cls"``, whose whole purpose is to reach a token that
          ``tokens`` would typically exclude.
        cls_index : int (default: 0)
          Index of the token to select when ``pooling="cls"``, applied to the
          model's full token sequence, before any ``tokens`` selection. Note
          that not all models place their class token first: Chronos-2, for
          instance, inserts a register token between the context and forecast
          tokens.
        input_layout : {"univariate", "channels_last", "channels_first"} (default: "univariate")
          How multivariate series are handed over to the model.
          ``"univariate"`` embeds every channel independently, as a
          ``(n_ts * d, sz)`` array, and concatenates the resulting per-channel
          embeddings, which is what most time series foundation models expect.
          The other two layouts feed a ``(n_ts, sz, d)`` or ``(n_ts, d, sz)``
          array respectively, for models that are natively multivariate.
        context_length : int or None (default: None)
          When set, only the last ``context_length`` timestamps of each series
          are fed to the model.
        input_name : str or None (default: None)
          Name of the ``forward`` argument receiving the context values. When
          None, it is auto-detected among common names.
        model_kwargs : dict or None (default: None)
          Extra keyword arguments passed to every ``forward`` call.
        batch_size : int (default: 32)
          Number of series embedded at once.
        device : str or None (default: None)
          Device on which inference is run, e.g. ``"cuda"``. When None, the
          device the model already lives on is used.
        verbose : int (default: 0)
          When positive, prints progress information.

    Attributes
    ----------
        n_features_in_ : int
          Number of features (channels) of the series seen during fit.
        embedding_size_ : int
          Dimension of the produced embeddings. When ``pooling`` is None, this
          is the number of features of each timestep of the produced series.
        n_tokens_ : int or None
          Number of tokens kept per series when ``pooling`` is None, that is,
          the length of the produced series. None otherwise.

    Notes
    -----
        This estimator does not support variable length time series, as the
        NaN padding used by tslearn to represent them would propagate through
        most models.

    Examples
    --------
    >>> from chronos import Chronos2Pipeline  # doctest: +SKIP
    >>> pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2")  # doctest: +SKIP
    >>> embedder = TimeSeriesFoundationEmbedder(pipeline.model, layer=-2)  # doctest: +SKIP
    >>> embedder.fit_transform(X).shape  # doctest: +SKIP
    (10, 512)

    Keeping one representation per token, as a time series dataset:

    >>> embedder = TimeSeriesFoundationEmbedder(  # doctest: +SKIP
    ...     model, layer=-2, pooling=None, tokens=(0, -2))
    >>> embedder.fit_transform(X).shape  # doctest: +SKIP
    (10, 16, 512)

    References
    ----------
    .. [1] G. Alain and Y. Bengio. Understanding intermediate layers using
      linear classifier probes. ICLR Workshop, 2017.

    """

    def __init__(
        self,
        model,
        layer=None,
        layers_path=None,
        pooling="mean",
        tokens=None,
        cls_index=0,
        input_layout="univariate",
        context_length=None,
        input_name=None,
        model_kwargs=None,
        batch_size=32,
        device=None,
        verbose=0,
    ):
        self.model = model
        self.layer = layer
        self.layers_path = layers_path
        self.pooling = pooling
        self.tokens = tokens
        self.cls_index = cls_index
        self.input_layout = input_layout
        self.context_length = context_length
        self.input_name = input_name
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def _validate_params_(self):
        if self.pooling not in POOLINGS:
            raise ValueError(
                f"`pooling` must be one of {POOLINGS}, got '{self.pooling}'."
            )
        _token_slice(self.tokens)
        if self.input_layout not in LAYOUTS:
            raise ValueError(
                f"`input_layout` must be one of {LAYOUTS}, got "
                f"'{self.input_layout}'."
            )
        _require_torch()
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(
                "`model` must be a torch.nn.Module, got "
                f"{type(self.model)}. Pre-trained models are typically obtained "
                "through a `from_pretrained` call."
            )

    @property
    def _device(self):
        if self.device is not None:
            return torch.device(self.device)
        try:
            return next(self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            return torch.device("cpu")

    def _resolve_input_name(self):
        if self.input_name is not None:
            return self.input_name
        try:
            signature = inspect.signature(self.model.forward)
        except (TypeError, ValueError):  # pragma: no cover
            return CANDIDATE_INPUT_NAMES[0]
        parameters = signature.parameters
        for name in CANDIDATE_INPUT_NAMES:
            if name in parameters:
                return name
        for name, parameter in parameters.items():
            if name == "self":
                continue
            if parameter.kind in (
                parameter.POSITIONAL_ONLY,
                parameter.POSITIONAL_OR_KEYWORD,
            ):
                return name
        raise ValueError(  # pragma: no cover
            "Could not determine which argument of the model's `forward` method "
            "should receive the time series. Pass an explicit `input_name`."
        )

    def _resolve_layers(self):
        if self.layers_path is not None:
            return self.layers_path, _resolve_attribute_path(
                self.model, self.layers_path
            )
        return _autodetect_layers(self.model)

    def _forward_hidden_states(self, batch):
        """Run the model on a batch and return hidden states of shape
        (batch, n_tokens, dim)."""
        kwargs = dict(self.model_kwargs or {})
        kwargs[self._input_name_] = batch

        if self.layer is None:
            with torch.no_grad():
                output = self.model(**kwargs)
            return _as_tensor(output)

        _, layers = self._resolve_layers()
        try:
            layer_module = layers[self.layer]
        except IndexError:
            raise ValueError(
                f"`layer={self.layer}` is out of range: the model's layer stack "
                f"holds {len(layers)} layers."
            )

        captured = {}

        def hook(_module, _inputs, output):
            captured["hidden_states"] = _as_tensor(output)

        handle = layer_module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.model(**kwargs)
        finally:
            handle.remove()

        if "hidden_states" not in captured:  # pragma: no cover
            raise RuntimeError(
                "The forward hook was never triggered; the selected layer does "
                "not seem to take part in the model's forward pass."
            )
        return captured["hidden_states"]

    def _pool(self, hidden_states):
        if hidden_states.ndim == 2:
            # Already pooled by the model itself
            return hidden_states
        if hidden_states.ndim != 3:
            raise ValueError(
                "Expected hidden states of shape (batch, n_tokens, dim), got "
                f"shape {tuple(hidden_states.shape)}."
            )
        pooling = _normalize_pooling(self.pooling)

        # The class token is deliberately read before any token selection, as
        # it is usually one of the tokens `tokens` is meant to filter out.
        if pooling == "cls":
            n_tokens = hidden_states.shape[1]
            if not -n_tokens <= self.cls_index < n_tokens:
                raise ValueError(
                    f"`cls_index={self.cls_index}` is out of range: the model "
                    f"produced {n_tokens} tokens."
                )
            return hidden_states[:, self.cls_index]

        hidden_states = hidden_states[:, _token_slice(self.tokens)]
        if hidden_states.shape[1] == 0:
            raise ValueError(
                f"`tokens={self.tokens!r}` selects no token at all out of the "
                "sequence produced by the model."
            )
        if pooling is None:
            return hidden_states
        if pooling == "mean":
            return hidden_states.mean(dim=1)
        if pooling == "max":
            return hidden_states.max(dim=1).values
        if pooling == "last":
            return hidden_states[:, -1]
        return hidden_states.reshape(hidden_states.shape[0], -1)

    def _check_input(self, X):
        X = check_array(X, allow_nd=True, force_all_finite=True)
        X = to_time_series_dataset(X)
        if self.context_length is not None:
            if X.shape[1] < self.context_length:
                raise ValueError(
                    f"Series of length at least context_length="
                    f"{self.context_length} are required, got {X.shape[1]}."
                )
            X = X[:, -self.context_length :]
        return X

    def _embed(self, X):
        """Embed a checked (n_ts, sz, d) dataset.

        Returns a ``(n_ts, D)`` feature matrix, or a ``(n_ts, n_tokens, D)``
        time series dataset when ``pooling`` is None.
        """
        n_ts, sz, d = X.shape
        flat = _layout_to_model_input(X, self.input_layout)

        embeddings = []
        device = self._device
        for start in range(0, flat.shape[0], self.batch_size):
            chunk = flat[start : start + self.batch_size]
            batch = torch.as_tensor(np.asarray(chunk, dtype=np.float32), device=device)
            hidden_states = self._forward_hidden_states(batch)
            embeddings.append(self._pool(hidden_states).to(torch.float32).cpu().numpy())
            if self.verbose:
                print(
                    f"Embedded {min(start + self.batch_size, flat.shape[0])}"
                    f"/{flat.shape[0]} series"
                )
        embeddings = np.concatenate(embeddings, axis=0)

        if self.input_layout == "univariate":
            if embeddings.ndim == 3:
                # (n_ts * d, n_tokens, dim) -> (n_ts, n_tokens, d * dim), so
                # that the per-channel representations of a same timestep sit
                # side by side, as in any other tslearn time series dataset.
                n_tokens, dim = embeddings.shape[1:]
                embeddings = embeddings.reshape(n_ts, d, n_tokens, dim)
                embeddings = np.swapaxes(embeddings, 1, 2)
                embeddings = embeddings.reshape(n_ts, n_tokens, d * dim)
            else:
                # Concatenate the per-channel embeddings of a same series
                embeddings = embeddings.reshape(n_ts, -1)
        return embeddings

    def fit(self, X, y=None):
        """Check the model and the input data. No parameter is learnt.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.
            y : Ignored

        Returns
        -------
            self
                The fitted estimator

        """
        self._validate_params_()
        X = self._check_input(X)
        self.model.eval()
        self._input_name_ = self._resolve_input_name()
        self.n_features_in_ = X.shape[2]
        self._sz_fit_ = X.shape[1]
        # A single series is enough to know the shape of the representations
        embedded = self._embed(X[:1])
        self.embedding_size_ = embedded.shape[-1]
        self.n_tokens_ = embedded.shape[1] if embedded.ndim == 3 else None
        return self

    def transform(self, X):
        """Embed a time series dataset.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.

        Returns
        -------
            array of shape=(n_ts, embedding_size_), or
            (n_ts, n_tokens_, embedding_size_) when ``pooling`` is None
              Frozen representations of the input series.

        """
        check_is_fitted(self, "embedding_size_")
        X = self._check_input(X)
        if X.shape[2] != self.n_features_in_:
            raise ValueError(
                f"Series with {self.n_features_in_} features were expected, got "
                f"{X.shape[2]}."
            )
        if self.pooling == "flatten" and X.shape[1] != self._sz_fit_:
            warnings.warn(
                "pooling='flatten' produces context-length dependent features; "
                f"series of length {self._sz_fit_} were seen at fit time and "
                f"series of length {X.shape[1]} are being transformed."
            )
        return self._embed(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the estimator and embed a time series dataset.

        Parameters
        ----------
            X : array-like of shape=(n_ts, sz, d)
                Time series dataset.
            y : Ignored

        Returns
        -------
            array of shape=(n_ts, embedding_size_), or
            (n_ts, n_tokens_, embedding_size_) when ``pooling`` is None
              Frozen representations of the input series.

        """
        return self.fit(X, y).transform(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.non_deterministic = False
        tags.input_tags.allow_nan = False
        return tags
