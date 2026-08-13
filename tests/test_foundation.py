"""Tests for the :mod:`tslearn.foundation` module.

These tests run entirely offline, against small hand-written models that
reproduce the calling conventions of the pre-trained models published on the
Hugging Face Hub, so that no download is ever needed.
"""

import numpy as np

import pytest

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from tslearn.generators import random_walks

torch = pytest.importorskip("torch")

from tslearn.foundation import (  # noqa: E402
    LinearProbeClassifier,
    LinearProbeForecaster,
    TimeSeriesFoundationEmbedder,
    ZeroShotForecaster,
)


PATCH_SIZE = 4
D_MODEL = 8


class _Block(torch.nn.Module):
    """A single, deliberately simple, encoder block."""

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, d_model)

    def forward(self, hidden_states):
        return hidden_states + torch.tanh(self.linear(hidden_states))


class _Encoder(torch.nn.Module):
    def __init__(self, n_layers=3, d_model=D_MODEL):
        super().__init__()
        self.block = torch.nn.ModuleList([_Block(d_model) for _ in range(n_layers)])
        self.final_layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, hidden_states):
        for block in self.block:
            hidden_states = block(hidden_states)
        return self.final_layer_norm(hidden_states)


class _Output:
    """Stands in for a ``transformers`` ``ModelOutput``."""

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _DummyBackbone(torch.nn.Module):
    """A patch-based encoder taking univariate series, like Chronos-2.

    A register token is appended after the context tokens, so that the
    ``pooling="token"`` code path can be exercised on a non-zero ``token_index``.
    """

    def __init__(self, n_layers=3, d_model=D_MODEL, patch_size=PATCH_SIZE, seed=0):
        super().__init__()
        # Seeded so that results do not depend on which test ran before
        torch.manual_seed(seed)
        self.patch_size = patch_size
        self.embedding = torch.nn.Linear(patch_size, d_model)
        self.register_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
        self.encoder = _Encoder(n_layers, d_model)

    def forward(self, context):
        batch_size, sz = context.shape
        n_patches = sz // self.patch_size
        patches = context[:, : n_patches * self.patch_size]
        patches = patches.reshape(batch_size, n_patches, self.patch_size)
        hidden_states = self.embedding(patches)
        register = self.register_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat([hidden_states, register], dim=1)
        return _Output(self.encoder(hidden_states))


class _MultivariateBackbone(torch.nn.Module):
    """An encoder taking (batch, sz, d) arrays, like ``transformers`` models."""

    def __init__(self, n_channels, d_model=D_MODEL, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = torch.nn.Linear(n_channels, d_model)
        self.encoder = _Encoder(2, d_model)

    def forward(self, past_values):
        return _Output(self.encoder(self.embedding(past_values)))


class _DummyPipeline:
    """A zero-shot forecaster mimicking ``Chronos2Pipeline``.

    It returns one tensor per series, of shape
    ``(n_variates, prediction_length, n_quantiles)``.
    """

    n_quantiles = 9

    def predict(self, inputs, prediction_length=1, batch_size=256):
        inputs = np.asarray(inputs, dtype=np.float64)
        # Naive forecast: repeat the last observed value
        last = inputs[:, -1]
        quantile_offsets = np.linspace(-1.0, 1.0, self.n_quantiles)
        forecast = (
            last[:, None, None]
            + np.zeros((1, prediction_length, 1))
            + quantile_offsets[None, None, :]
        )
        return [torch.as_tensor(row[None]) for row in forecast]


def _dataset(n_ts=6, sz=32, d=1, seed=0):
    return random_walks(n_ts=n_ts, sz=sz, d=d, random_state=seed)


# ---------------------------------------------------------------------------
# TimeSeriesFoundationEmbedder
# ---------------------------------------------------------------------------


def test_embedder_shapes_and_layouts():
    X = _dataset(n_ts=5, sz=32, d=1)
    embedder = TimeSeriesFoundationEmbedder(_DummyBackbone())
    embeddings = embedder.fit_transform(X)
    assert embeddings.shape == (5, D_MODEL)
    assert embedder.embedding_size_ == D_MODEL

    # Multivariate series are embedded channel per channel and concatenated
    X_multi = _dataset(n_ts=5, sz=32, d=3)
    embeddings = TimeSeriesFoundationEmbedder(_DummyBackbone()).fit_transform(X_multi)
    assert embeddings.shape == (5, 3 * D_MODEL)

    # Natively multivariate models get the whole (n_ts, sz, d) array
    embedder = TimeSeriesFoundationEmbedder(
        _MultivariateBackbone(n_channels=3), input_layout="channels_last"
    )
    assert embedder.fit_transform(X_multi).shape == (5, D_MODEL)


def test_embedder_channel_stacking_is_order_preserving():
    # The embedding of a multivariate series must be the concatenation of the
    # embeddings of its channels, in channel order.
    X = _dataset(n_ts=4, sz=32, d=2)
    backbone = _DummyBackbone()
    joint = TimeSeriesFoundationEmbedder(backbone).fit_transform(X)
    per_channel = [
        TimeSeriesFoundationEmbedder(backbone).fit_transform(X[:, :, k : k + 1])
        for k in range(2)
    ]
    np.testing.assert_allclose(
        joint, np.concatenate(per_channel, axis=1), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("pooling", ["mean", "max", "token", "last", "flatten"])
def test_embedder_poolings(pooling):
    X = _dataset(n_ts=4, sz=32, d=1)
    embedder = TimeSeriesFoundationEmbedder(_DummyBackbone(), pooling=pooling)
    embeddings = embedder.fit_transform(X)
    n_tokens = 32 // PATCH_SIZE + 1  # context patches + register token
    expected = n_tokens * D_MODEL if pooling == "flatten" else D_MODEL
    assert embeddings.shape == (4, expected)
    assert embedder.n_tokens_ is None


@pytest.mark.parametrize("pooling", [None, "none"])
def test_embedder_without_pooling_is_a_series_to_series_transform(pooling):
    X = _dataset(n_ts=4, sz=32, d=1)
    n_tokens = 32 // PATCH_SIZE + 1  # context patches + register token

    embedder = TimeSeriesFoundationEmbedder(_DummyBackbone(), pooling=pooling)
    embeddings = embedder.fit_transform(X)
    assert embeddings.shape == (4, n_tokens, D_MODEL)
    assert embedder.n_tokens_ == n_tokens
    assert embedder.embedding_size_ == D_MODEL

    # The output is a valid tslearn dataset, usable by other estimators
    from tslearn.clustering import TimeSeriesKMeans

    labels = TimeSeriesKMeans(n_clusters=2, max_iter=2, random_state=0).fit_predict(
        embeddings
    )
    assert labels.shape == (4,)


def test_embedder_token_selection():
    X = _dataset(n_ts=4, sz=32, d=1)
    n_context = 32 // PATCH_SIZE  # tokens that represent the series
    backbone = _DummyBackbone()

    # Dropping the trailing register token leaves the context tokens
    embeddings = TimeSeriesFoundationEmbedder(
        backbone, pooling=None, tokens=(0, -1)
    ).fit_transform(X)
    assert embeddings.shape == (4, n_context, D_MODEL)

    # Equivalent spellings
    np.testing.assert_allclose(
        embeddings,
        TimeSeriesFoundationEmbedder(
            backbone, pooling=None, tokens=slice(0, -1)
        ).fit_transform(X),
    )
    np.testing.assert_allclose(
        embeddings,
        TimeSeriesFoundationEmbedder(
            backbone, pooling=None, tokens=(None, n_context)
        ).fit_transform(X),
    )

    # Token selection also applies before pooling, and changes the result:
    # averaging over the context tokens is not averaging over all of them
    without_register = TimeSeriesFoundationEmbedder(
        backbone, pooling="mean", tokens=(0, -1)
    ).fit_transform(X)
    over_everything = TimeSeriesFoundationEmbedder(
        backbone, pooling="mean"
    ).fit_transform(X)
    np.testing.assert_allclose(
        without_register, embeddings.mean(axis=1), rtol=1e-5, atol=1e-6
    )
    assert not np.allclose(without_register, over_everything)

    # ... but not to "token", which is meant to reach an excluded token
    np.testing.assert_allclose(
        TimeSeriesFoundationEmbedder(
            backbone, pooling="token", token_index=-1, tokens=(0, -1)
        ).fit_transform(X),
        TimeSeriesFoundationEmbedder(
            backbone, pooling="token", token_index=-1
        ).fit_transform(X),
    )


def test_embedder_token_selection_multivariate():
    X = _dataset(n_ts=4, sz=32, d=3)
    n_context = 32 // PATCH_SIZE
    embeddings = TimeSeriesFoundationEmbedder(
        _DummyBackbone(), pooling=None, tokens=(0, -1)
    ).fit_transform(X)
    # Channels are concatenated along the feature axis, one row per token
    assert embeddings.shape == (4, n_context, 3 * D_MODEL)

    # Channel k of the output must be the univariate embedding of channel k
    backbone = _DummyBackbone()
    joint = TimeSeriesFoundationEmbedder(
        backbone, pooling=None, tokens=(0, -1)
    ).fit_transform(X)
    for k in range(3):
        alone = TimeSeriesFoundationEmbedder(
            backbone, pooling=None, tokens=(0, -1)
        ).fit_transform(X[:, :, k : k + 1])
        np.testing.assert_allclose(
            joint[:, :, k * D_MODEL : (k + 1) * D_MODEL],
            alone,
            rtol=1e-5,
            atol=1e-6,
        )


def test_embedder_token_selection_errors():
    X = _dataset(n_ts=4, sz=32, d=1)
    with pytest.raises(ValueError, match="`tokens` must be"):
        TimeSeriesFoundationEmbedder(_DummyBackbone(), tokens=3).fit(X)
    with pytest.raises(ValueError, match="`tokens` must be"):
        TimeSeriesFoundationEmbedder(_DummyBackbone(), tokens=(1, 2, 3)).fit(X)
    with pytest.raises(ValueError, match="selects no token"):
        TimeSeriesFoundationEmbedder(_DummyBackbone(), tokens=(0, 0)).fit(X)


def test_probes_reject_unpooled_representations():
    X, y = _classification_dataset()
    for pooling in (None, "none"):
        with pytest.raises(ValueError, match="pooling=None"):
            LinearProbeClassifier(_DummyBackbone(), pooling=pooling).fit(X, y)
        with pytest.raises(ValueError, match="pooling=None"):
            LinearProbeForecaster(
                _DummyBackbone(), pooling=pooling, context_length=16, horizon=2
            ).fit(X)


def test_probes_accept_token_selection():
    X, y = _classification_dataset()
    model = LinearProbeClassifier(_DummyBackbone(), tokens=(0, -1)).fit(X, y)
    assert model.predict(X).shape == (len(y),)

    X_fc = _dataset(n_ts=6, sz=48, d=1)
    model = LinearProbeForecaster(
        _DummyBackbone(), context_length=16, horizon=2, stride=8, tokens=(0, -1)
    ).fit(X_fc)
    assert model.predict(X_fc).shape == (6, 2, 1)


def test_embedder_token_index_selects_the_register_token():
    X = _dataset(n_ts=3, sz=32, d=1)
    # The register token is appended last and does not depend on the input, so
    # selecting it must yield identical embeddings for all series.
    embeddings = TimeSeriesFoundationEmbedder(
        _DummyBackbone(), pooling="token", token_index=-1, layer=0
    ).fit_transform(X)
    np.testing.assert_allclose(
        embeddings, np.repeat(embeddings[:1], 3, axis=0), rtol=1e-5, atol=1e-6
    )

    # Whereas a context token does depend on the input
    embeddings = TimeSeriesFoundationEmbedder(
        _DummyBackbone(), pooling="token", token_index=0, layer=0
    ).fit_transform(X)
    assert not np.allclose(embeddings, embeddings[:1])


def test_embedder_layer_selection():
    X = _dataset(n_ts=4, sz=32, d=1)
    backbone = _DummyBackbone(n_layers=3)
    embeddings = [
        TimeSeriesFoundationEmbedder(backbone, layer=layer).fit_transform(X)
        for layer in (0, 1, 2, None)
    ]
    # Every layer yields a different representation of the same data
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            assert not np.allclose(embeddings[i], embeddings[j])

    # Negative indices address the stack from the end
    np.testing.assert_allclose(
        TimeSeriesFoundationEmbedder(backbone, layer=-1).fit_transform(X),
        embeddings[2],
        rtol=1e-5,
        atol=1e-6,
    )

    # An explicit path to the layer stack gives the same result as autodetection
    np.testing.assert_allclose(
        TimeSeriesFoundationEmbedder(
            backbone, layer=1, layers_path="encoder.block"
        ).fit_transform(X),
        embeddings[1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_embedder_leaves_the_model_frozen():
    X = _dataset(n_ts=4, sz=32, d=1)
    backbone = _DummyBackbone()
    before = [p.detach().clone() for p in backbone.parameters()]
    TimeSeriesFoundationEmbedder(backbone).fit_transform(X)
    for parameter, reference in zip(backbone.parameters(), before):
        assert parameter.grad is None
        torch.testing.assert_close(parameter.detach(), reference)


def test_embedder_context_length_truncation():
    X = _dataset(n_ts=4, sz=64, d=1)
    embedder = TimeSeriesFoundationEmbedder(_DummyBackbone(), context_length=32)
    np.testing.assert_allclose(
        embedder.fit_transform(X),
        TimeSeriesFoundationEmbedder(embedder.model).fit_transform(X[:, -32:]),
        rtol=1e-5,
        atol=1e-6,
    )

    with pytest.raises(ValueError, match="context_length"):
        TimeSeriesFoundationEmbedder(
            _DummyBackbone(), context_length=128
        ).fit_transform(X)


def test_embedder_errors():
    X = _dataset(n_ts=4, sz=32, d=1)
    with pytest.raises(ValueError, match="pooling"):
        TimeSeriesFoundationEmbedder(_DummyBackbone(), pooling="median").fit(X)
    with pytest.raises(ValueError, match="input_layout"):
        TimeSeriesFoundationEmbedder(_DummyBackbone(), input_layout="nchw").fit(X)
    with pytest.raises(TypeError, match="torch.nn.Module"):
        TimeSeriesFoundationEmbedder(_DummyPipeline()).fit(X)
    with pytest.raises(ValueError, match="out of range"):
        TimeSeriesFoundationEmbedder(_DummyBackbone(n_layers=3), layer=7).fit(X)
    with pytest.raises(ValueError, match="token_index"):
        TimeSeriesFoundationEmbedder(
            _DummyBackbone(), pooling="token", token_index=999
        ).fit(X)
    with pytest.raises(ValueError, match="features"):
        TimeSeriesFoundationEmbedder(_DummyBackbone()).fit(X).transform(
            _dataset(n_ts=4, sz=32, d=2)
        )


def test_embedder_in_a_sklearn_pipeline():
    X = _dataset(n_ts=20, sz=32, d=1)
    y = np.arange(20) % 2
    pipeline = make_pipeline(
        TimeSeriesFoundationEmbedder(_DummyBackbone()), LinearSVC()
    )
    assert pipeline.fit(X, y).predict(X).shape == (20,)
    assert clone(pipeline) is not pipeline


def test_embedder_channels_first_layout():
    X = _dataset(n_ts=4, sz=32, d=3)
    seen = {}

    class _ChannelsFirstBackbone(torch.nn.Module):
        """Expects a (batch, d, sz) input, like some ``transformers`` models."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, D_MODEL)

        def forward(self, series):
            seen["shape"] = tuple(series.shape)
            return _Output(self.linear(series))

    embedder = TimeSeriesFoundationEmbedder(
        _ChannelsFirstBackbone(), input_layout="channels_first", pooling="mean"
    )
    embeddings = embedder.fit_transform(X)
    assert seen["shape"] == (4, 3, 32)
    assert embeddings.shape == (4, D_MODEL)


def test_embedder_layers_path_with_numeric_segment():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _WrappedBackbone(torch.nn.Module):
        """Nests the encoder in a top-level ``ModuleList``, as some models do,
        requiring a numeric index to reach the stack of layers."""

        def __init__(self, n_layers=3, d_model=D_MODEL, patch_size=PATCH_SIZE, seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.patch_size = patch_size
            self.embedding = torch.nn.Linear(patch_size, d_model)
            self.register_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
            self.stages = torch.nn.ModuleList([_Encoder(n_layers, d_model)])

        def forward(self, context):
            batch_size, sz = context.shape
            n_patches = sz // self.patch_size
            patches = context[:, : n_patches * self.patch_size]
            patches = patches.reshape(batch_size, n_patches, self.patch_size)
            hidden_states = self.embedding(patches)
            register = self.register_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat([hidden_states, register], dim=1)
            return _Output(self.stages[0](hidden_states))

    backbone = _WrappedBackbone()
    embeddings = TimeSeriesFoundationEmbedder(
        backbone, layer=1, layers_path="stages.0.block"
    ).fit_transform(X)
    assert embeddings.shape == (3, D_MODEL)

    with pytest.raises(AttributeError, match="Could not resolve"):
        TimeSeriesFoundationEmbedder(
            backbone, layer=0, layers_path="stages.0.bogus"
        ).fit(X)


def test_embedder_autodetect_layers_failure():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _NoLayerStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, D_MODEL)

        def forward(self, x):
            return _Output(self.linear(x).unsqueeze(1))

    with pytest.raises(ValueError, match="Could not automatically locate"):
        TimeSeriesFoundationEmbedder(_NoLayerStack(), layer=0).fit(X)


def test_embedder_accepts_dict_model_outputs():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _DictOutputBackbone(torch.nn.Module):
        """Returns a plain dict, as some pipelines do instead of a ModelOutput."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, D_MODEL)

        def forward(self, x):
            return {"hidden_states": self.linear(x).unsqueeze(1)}

    embeddings = TimeSeriesFoundationEmbedder(_DictOutputBackbone()).fit_transform(X)
    assert embeddings.shape == (3, D_MODEL)


def test_embedder_accepts_hidden_states_tuple_attribute():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _LayerStackOutput:
        """Stands in for a ``transformers`` output exposing all-layer states."""

        def __init__(self, layers):
            self.hidden_states = layers

    class _AllLayersBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.linear = torch.nn.Linear(32, D_MODEL)

        def forward(self, x):
            hidden = self.linear(x).unsqueeze(1)
            return _LayerStackOutput((hidden, 2 * hidden))

    backbone = _AllLayersBackbone()
    embeddings = TimeSeriesFoundationEmbedder(backbone).fit_transform(X)
    with torch.no_grad():
        raw = backbone.linear(torch.as_tensor(X[:, :, 0], dtype=torch.float32))
    # Only the last element of the `hidden_states` tuple is read, matching
    # transformers' convention that it holds one tensor per layer
    np.testing.assert_allclose(embeddings, 2 * raw.numpy(), rtol=1e-5, atol=1e-6)


def test_embedder_accepts_plain_tuple_outputs():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _PlainTupleBackbone(torch.nn.Module):
        """Returns a raw tuple, with no ``ModelOutput`` wrapper at all."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, D_MODEL)

        def forward(self, x):
            hidden = self.linear(x)
            # The 3d sequence of hidden states must be preferred over the
            # pooled 2d tensor that precedes it in the tuple
            return (hidden, hidden.unsqueeze(1))

    embeddings = TimeSeriesFoundationEmbedder(_PlainTupleBackbone()).fit_transform(X)
    assert embeddings.shape == (3, D_MODEL)


def test_embedder_plain_tuple_falls_back_to_any_tensor():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _NoSequenceBackbone(torch.nn.Module):
        """Returns an already-pooled 2d tensor wrapped in a 1-tuple."""

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, D_MODEL)

        def forward(self, x):
            return (self.linear(x),)

    embeddings = TimeSeriesFoundationEmbedder(_NoSequenceBackbone()).fit_transform(X)
    assert embeddings.shape == (3, D_MODEL)


def test_embedder_rejects_unrecognized_model_output():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _BogusOutputBackbone(torch.nn.Module):
        def forward(self, x):
            return "not a tensor"

    with pytest.raises(TypeError, match="Could not extract a hidden state"):
        TimeSeriesFoundationEmbedder(_BogusOutputBackbone()).fit(X)


def test_embedder_rejects_invalid_hidden_state_rank():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _WeirdRankBackbone(torch.nn.Module):
        def forward(self, x):
            return x[:, None, None, :]  # 4d, not a valid hidden-state rank

    with pytest.raises(ValueError, match="Expected hidden states of shape"):
        TimeSeriesFoundationEmbedder(_WeirdRankBackbone()).fit(X)


def test_embedder_explicit_device():
    X = _dataset(n_ts=3, sz=32, d=1)
    embeddings = TimeSeriesFoundationEmbedder(
        _DummyBackbone(), device="cpu"
    ).fit_transform(X)
    assert embeddings.shape == (3, D_MODEL)


def test_embedder_explicit_input_name():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _NamedArgBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, D_MODEL)

        def forward(self, my_custom_arg):
            return _Output(self.linear(my_custom_arg).unsqueeze(1))

    embedder = TimeSeriesFoundationEmbedder(
        _NamedArgBackbone(), input_name="my_custom_arg"
    )
    embeddings = embedder.fit_transform(X)
    assert embeddings.shape == (3, D_MODEL)


def test_embedder_resolves_input_name_by_position_when_unrecognized():
    X = _dataset(n_ts=3, sz=32, d=1)

    class _PositionalArgBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, D_MODEL)

        # `data` is not one of CANDIDATE_INPUT_NAMES, forcing the fallback to
        # the first positional parameter of `forward`
        def forward(self, data):
            return _Output(self.linear(data).unsqueeze(1))

    embedder = TimeSeriesFoundationEmbedder(_PositionalArgBackbone())
    embeddings = embedder.fit_transform(X)
    assert embedder._input_name_ == "data"
    assert embeddings.shape == (3, D_MODEL)


def test_embedder_verbose_prints_progress(capsys):
    X = _dataset(n_ts=5, sz=32, d=1)
    TimeSeriesFoundationEmbedder(
        _DummyBackbone(), batch_size=2, verbose=1
    ).fit_transform(X)
    captured = capsys.readouterr()
    assert "Embedded 5/5 series" in captured.out


def test_embedder_flatten_pooling_warns_on_length_mismatch():
    X_fit = _dataset(n_ts=3, sz=32, d=1)
    X_other = _dataset(n_ts=3, sz=48, d=1, seed=1)
    embedder = TimeSeriesFoundationEmbedder(
        _DummyBackbone(), pooling="flatten"
    ).fit(X_fit)
    with pytest.warns(UserWarning, match="pooling='flatten' produces context-length"):
        embedder.transform(X_other)


# ---------------------------------------------------------------------------
# ZeroShotForecaster
# ---------------------------------------------------------------------------


def test_zero_shot_forecaster():
    X = _dataset(n_ts=5, sz=32, d=1)
    model = ZeroShotForecaster(_DummyPipeline())

    with pytest.warns(UserWarning, match="does not train anything"):
        model.fit(X)

    predicted = model.predict(X, n=7)
    assert predicted.shape == (5, 7, 1)
    # The dummy pipeline repeats the last value, and the median quantile of its
    # symmetric output is that value exactly
    np.testing.assert_allclose(
        predicted, np.repeat(X[:, -1:], 7, axis=1), rtol=1e-5, atol=1e-6
    )


def test_zero_shot_forecaster_multivariate():
    X = _dataset(n_ts=5, sz=32, d=3)
    predicted = ZeroShotForecaster(_DummyPipeline()).predict(X, n=4)
    assert predicted.shape == (5, 4, 3)
    # Channels must not get mixed up while being folded in and out of the batch
    np.testing.assert_allclose(
        predicted, np.repeat(X[:, -1:], 4, axis=1), rtol=1e-5, atol=1e-6
    )


def test_zero_shot_forecaster_quantile_selection():
    X = _dataset(n_ts=3, sz=32, d=1)
    low = ZeroShotForecaster(_DummyPipeline(), quantile=0.1).predict(X, n=3)
    high = ZeroShotForecaster(_DummyPipeline(), quantile=0.9).predict(X, n=3)
    assert np.all(low < high)


def test_zero_shot_forecaster_custom_predict_fn():
    X = _dataset(n_ts=4, sz=32, d=1)

    def predict_fn(model, context, horizon):
        assert context.shape == (4, 32)
        return np.zeros((context.shape[0], horizon))

    predicted = ZeroShotForecaster(object(), predict_fn=predict_fn).predict(X, n=5)
    np.testing.assert_array_equal(predicted, np.zeros((4, 5, 1)))


def test_zero_shot_forecaster_context_length():
    X = _dataset(n_ts=4, sz=64, d=1)
    seen = {}

    def predict_fn(model, context, horizon):
        seen["shape"] = context.shape
        return np.zeros((context.shape[0], horizon))

    ZeroShotForecaster(
        object(), predict_fn=predict_fn, context_length=16
    ).predict(X, n=2)
    assert seen["shape"] == (4, 16)


def test_zero_shot_forecaster_negotiates_the_context_format():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _NeedsVariateAxis:
        """Rejects 2d contexts, like ``Chronos2Pipeline`` does."""

        def predict(self, inputs, prediction_length=1):
            inputs = np.asarray(inputs)
            if inputs.ndim != 3:
                raise ValueError(
                    "Expected 3-d tensor with shape "
                    "(n_series, n_variates, history_length)."
                )
            return np.zeros((inputs.shape[0], prediction_length))

    model = ZeroShotForecaster(_NeedsVariateAxis())
    assert model.predict(X, n=3).shape == (4, 3, 1)
    assert model.context_format_ == "3d"
    # The negotiated format is reused rather than re-discovered
    assert model.predict(X, n=3).shape == (4, 3, 1)

    class _NeedsList:
        def predict(self, inputs, prediction_length=1):
            if not isinstance(inputs, list):
                raise TypeError("A list of series is expected.")
            return np.zeros((len(inputs), prediction_length))

    model = ZeroShotForecaster(_NeedsList())
    assert model.predict(X, n=3).shape == (4, 3, 1)
    assert model.context_format_ == "list"

    class _RejectsEverything:
        def predict(self, inputs, prediction_length=1):
            raise ValueError("nope")

    with pytest.raises(ValueError, match="rejected every supported way"):
        ZeroShotForecaster(_RejectsEverything()).predict(X, n=3)


def test_zero_shot_forecaster_explicit_horizon_axis():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _HorizonFirst:
        """Returns (n_series, horizon, n_quantiles), like predict_quantiles."""

        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            levels = np.arange(5)[None, None, :]
            return np.zeros((n_rows, prediction_length, 5)) + levels

    class _HorizonLast:
        """Returns (n_series, n_quantiles, horizon), like Chronos-2."""

        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            levels = np.arange(5)[None, :, None]
            return np.zeros((n_rows, 5, prediction_length)) + levels

    # Both conventions are handled once the horizon axis is stated
    for model, axis in [(_HorizonFirst(), 1), (_HorizonLast(), 2)]:
        predicted = ZeroShotForecaster(model, horizon_axis=axis).predict(X, n=5)
        np.testing.assert_allclose(predicted, np.full((4, 5, 1), 2.0))

    # Negative indices are accepted
    np.testing.assert_allclose(
        ZeroShotForecaster(_HorizonLast(), horizon_axis=-1).predict(X, n=5),
        np.full((4, 5, 1), 2.0),
    )

    # A wrong axis is reported rather than silently mis-slicing
    with pytest.raises(ValueError, match="cannot hold a horizon"):
        ZeroShotForecaster(_HorizonLast(), horizon_axis=1).predict(X, n=7)
    with pytest.raises(ValueError, match="out of range"):
        ZeroShotForecaster(_HorizonLast(), horizon_axis=5).predict(X, n=3)
    with pytest.raises(ValueError, match="out of range"):
        ZeroShotForecaster(_HorizonLast(), horizon_axis=0).predict(X, n=3)
    with pytest.raises(ValueError, match="must be an integer or 'auto'"):
        ZeroShotForecaster(_HorizonLast(), horizon_axis="last").predict(X, n=3)


def test_zero_shot_forecaster_explicit_horizon_axis_lifts_ambiguity():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _Ambiguous:
        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            # (n_series, horizon, horizon), the second axis being the horizon
            steps = np.arange(prediction_length)[None, :, None]
            return np.zeros((n_rows, prediction_length, prediction_length)) + steps

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predicted = ZeroShotForecaster(_Ambiguous(), horizon_axis=1).predict(X, n=4)
    np.testing.assert_allclose(
        predicted, np.tile(np.arange(4.0)[None, :, None], (4, 1, 1))
    )


def test_zero_shot_forecaster_warns_on_ambiguous_output_shape():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _Ambiguous:
        """Returns as many quantiles as there are forecast steps."""

        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            return np.zeros((n_rows, prediction_length, prediction_length))

    with pytest.warns(UserWarning, match="Set `horizon_axis` explicitly"):
        ZeroShotForecaster(_Ambiguous()).predict(X, n=5)

    # A horizon of one is not ambiguous, singleton axes are simply reduced
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ZeroShotForecaster(_Ambiguous()).predict(X, n=1).shape == (4, 1, 1)


def test_zero_shot_forecaster_reduces_quantiles_to_a_point_forecast():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _Quantiles:
        """Returns (n_series, n_quantiles, horizon), like Chronos-2."""

        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            levels = np.arange(9)[None, :, None]
            return np.zeros((n_rows, 9, prediction_length)) + levels

    predicted = ZeroShotForecaster(_Quantiles()).predict(X, n=4)
    # The median of 0..8 is 4, whatever the horizon
    np.testing.assert_allclose(predicted, np.full((4, 4, 1), 4.0))
    np.testing.assert_allclose(
        ZeroShotForecaster(_Quantiles(), quantile=None).predict(X, n=4),
        np.full((4, 4, 1), 4.0),
    )
    np.testing.assert_allclose(
        ZeroShotForecaster(_Quantiles(), quantile=0.0).predict(X, n=4),
        np.zeros((4, 4, 1)),
    )


def test_zero_shot_forecaster_errors():
    X = _dataset(n_ts=4, sz=32, d=1)
    with pytest.raises(ValueError, match="`X` is required"):
        ZeroShotForecaster(_DummyPipeline()).predict(None)
    with pytest.raises(ValueError, match="positive integer"):
        ZeroShotForecaster(_DummyPipeline()).predict(X, n=0)
    with pytest.raises(ValueError, match="forecasting method"):
        ZeroShotForecaster(object()).predict(X, n=1)
    with pytest.raises(ValueError, match="input_layout"):
        ZeroShotForecaster(_DummyPipeline(), input_layout="bogus").predict(X)


def test_zero_shot_forecaster_skips_methods_without_a_horizon_argument():
    X = _dataset(n_ts=4, sz=32, d=1)
    calls = []

    class _PartialModel:
        """Exposes a `predict` with no forecast-horizon argument, which must
        be skipped in favor of `forecast`."""

        def predict(self, inputs):
            calls.append("predict")
            raise AssertionError("predict should never be called")

        def forecast(self, inputs, prediction_length=1):
            calls.append("forecast")
            n_rows = len(inputs)
            return np.zeros((n_rows, prediction_length))

    predicted = ZeroShotForecaster(_PartialModel()).predict(X, n=3)
    assert predicted.shape == (4, 3, 1)
    assert calls == ["forecast"]


def test_zero_shot_forecaster_accepts_object_attribute_output():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _AttributeOutput:
        """Stands in for an output object exposing the forecast as an attribute."""

        def __init__(self, mean):
            self.mean = mean

    class _ObjectOutputModel:
        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            return _AttributeOutput(np.zeros((n_rows, prediction_length)))

    predicted = ZeroShotForecaster(_ObjectOutputModel()).predict(X, n=3)
    assert predicted.shape == (4, 3, 1)


def test_zero_shot_forecaster_univariate_row_mismatch_error():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _WrongRowCountModel:
        def predict(self, inputs, prediction_length=1):
            return np.zeros((len(inputs) - 1, prediction_length))

    with pytest.raises(
        ValueError, match="forecasts for 3 series while 4 were provided"
    ):
        ZeroShotForecaster(_WrongRowCountModel()).predict(X, n=2)


def test_zero_shot_forecaster_univariate_forecast_too_short_error():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _TooShortModel:
        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            return np.zeros((n_rows, prediction_length - 1))  # one step short

    with pytest.raises(ValueError, match="shorter than the requested horizon"):
        ZeroShotForecaster(_TooShortModel()).predict(X, n=3)


def test_zero_shot_forecaster_accepts_dict_output():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _DictOutputModel:
        """Returns a plain dict, as some pipelines do instead of a ModelOutput."""

        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            return {"predictions": np.zeros((n_rows, prediction_length))}

    predicted = ZeroShotForecaster(_DictOutputModel()).predict(X, n=3)
    assert predicted.shape == (4, 3, 1)


def test_zero_shot_forecaster_empty_output_error():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _EmptyOutputModel:
        def predict(self, inputs, prediction_length=1):
            return []

    with pytest.raises(ValueError, match="empty forecast"):
        ZeroShotForecaster(_EmptyOutputModel()).predict(X, n=3)


def test_zero_shot_forecaster_heterogeneous_output_keeps_first_entry():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _HeterogeneousOutputModel:
        """Returns (forecast, extra_diagnostics), as some pipelines do."""

        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            forecast = np.zeros((n_rows, prediction_length))
            diagnostics = np.zeros((n_rows, 2))  # a different shape
            return (forecast, diagnostics)

    predicted = ZeroShotForecaster(_HeterogeneousOutputModel()).predict(X, n=3)
    assert predicted.shape == (4, 3, 1)


def test_zero_shot_forecaster_horizon_axis_prefers_smallest_longer_axis():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _PaddedForecastModel:
        """Pads the forecast length to a fixed block size larger than asked,
        with no axis exactly matching the horizon."""

        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            n_samples = 3  # sample paths, always shorter than the horizon here
            padded_length = prediction_length + 5
            return np.zeros((n_rows, n_samples, padded_length))

    predicted = ZeroShotForecaster(_PaddedForecastModel()).predict(X, n=4)
    np.testing.assert_allclose(predicted, np.zeros((4, 4, 1)))


def test_zero_shot_forecaster_horizon_axis_cannot_be_identified():
    X = _dataset(n_ts=4, sz=32, d=1)

    class _TooShortEverywhereModel:
        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            return np.zeros((n_rows, 2, 2))

    with pytest.raises(
        ValueError, match="Could not identify the forecast horizon axis"
    ):
        ZeroShotForecaster(_TooShortEverywhereModel()).predict(X, n=4)


def test_zero_shot_forecaster_natively_multivariate_channels_last():
    X = _dataset(n_ts=4, sz=32, d=3)

    class _NativelyMultivariateModel:
        """Returns per-channel forecasts directly, given a (n_ts, sz, d) input."""

        def predict(self, inputs, prediction_length=1):
            inputs = np.asarray(inputs)
            last = inputs[:, -1, :]  # (n_ts, d)
            return np.repeat(last[:, None, :], prediction_length, axis=1)

    predicted = ZeroShotForecaster(
        _NativelyMultivariateModel(), input_layout="channels_last"
    ).predict(X, n=4)
    assert predicted.shape == (4, 4, 3)
    np.testing.assert_allclose(predicted, np.repeat(X[:, -1:], 4, axis=1))


def test_zero_shot_forecaster_natively_multivariate_channels_first():
    X = _dataset(n_ts=4, sz=32, d=3)

    class _ChannelsFirstModel:
        """Expects a (n_ts, d, sz) input and returns (n_ts, d, horizon)."""

        def predict(self, inputs, prediction_length=1):
            inputs = np.asarray(inputs)
            last = inputs[:, :, -1]  # (n_ts, d)
            return np.repeat(last[:, :, None], prediction_length, axis=2)

    predicted = ZeroShotForecaster(
        _ChannelsFirstModel(), input_layout="channels_first"
    ).predict(X, n=4)
    assert predicted.shape == (4, 4, 3)
    np.testing.assert_allclose(predicted, np.repeat(X[:, -1:], 4, axis=1))


def test_zero_shot_forecaster_natively_multivariate_errors():
    X = _dataset(n_ts=4, sz=32, d=3)

    class _WrongBatchModel:
        def predict(self, inputs, prediction_length=1):
            return np.zeros((len(inputs) - 1, prediction_length, 3))

    with pytest.raises(
        ValueError, match="forecasts for 3 series while 4 were provided"
    ):
        ZeroShotForecaster(
            _WrongBatchModel(), input_layout="channels_last"
        ).predict(X, n=2)

    class _WrongChannelsModel:
        def predict(self, inputs, prediction_length=1):
            n_rows = len(inputs)
            return np.zeros((n_rows, prediction_length, 5))  # wrong d

    with pytest.raises(ValueError, match="Expected forecasts of shape"):
        ZeroShotForecaster(
            _WrongChannelsModel(), input_layout="channels_last"
        ).predict(X, n=2)


# ---------------------------------------------------------------------------
# LinearProbeForecaster
# ---------------------------------------------------------------------------


def test_linear_probe_forecaster():
    X = _dataset(n_ts=8, sz=64, d=1)
    model = LinearProbeForecaster(
        _DummyBackbone(), context_length=16, horizon=4, stride=4
    )
    model.fit(X)
    # 8 series x windows of span 20 taken every 4 timestamps
    assert model.n_windows_ == 8 * len(range(0, 64 - 20 + 1, 4))
    predicted = model.predict(X)
    assert predicted.shape == (8, 4, 1)
    # A shorter horizon truncates the forecast rather than refitting
    np.testing.assert_allclose(model.predict(X, n=2), predicted[:, :2])


def test_linear_probe_forecaster_multivariate():
    X = _dataset(n_ts=6, sz=48, d=2)
    model = LinearProbeForecaster(
        _DummyBackbone(), context_length=16, horizon=3, stride=8
    ).fit(X)
    assert model.predict(X).shape == (6, 3, 2)
    assert model.embedder_.embedding_size_ == 2 * D_MODEL


def test_linear_probe_forecaster_learns_something():
    # On a dataset of pure sine waves with varying phases, a probe on top of a
    # frozen encoder should do clearly better than forecasting the last value.
    rng = np.random.RandomState(0)
    t = np.linspace(0, 8 * np.pi, 96)
    phases = rng.uniform(0, 2 * np.pi, size=40)
    X = np.sin(t[None, :] + phases[:, None])[:, :, None]

    model = LinearProbeForecaster(
        _DummyBackbone(), context_length=16, horizon=4, stride=2
    ).fit(X[:30])
    assert model.score(X[30:]) > 0.5


def test_linear_probe_forecaster_accepts_any_probe():
    X = _dataset(n_ts=6, sz=48, d=1)
    model = LinearProbeForecaster(
        _DummyBackbone(),
        probe=Ridge(alpha=1.0),
        context_length=16,
        horizon=2,
        stride=8,
    ).fit(X)
    assert isinstance(model.probe_, Ridge)
    # The passed estimator is cloned, not fitted in place
    assert model.probe_ is not model.probe
    assert not hasattr(model.probe, "coef_")


def test_linear_probe_forecaster_errors():
    X = _dataset(n_ts=4, sz=48, d=1)
    with pytest.raises(ValueError, match="context_length \\+ horizon"):
        LinearProbeForecaster(
            _DummyBackbone(), context_length=40, horizon=16
        ).fit(X)
    with pytest.raises(ValueError, match="positive integer"):
        LinearProbeForecaster(_DummyBackbone(), horizon=0).fit(X)

    model = LinearProbeForecaster(
        _DummyBackbone(), context_length=16, horizon=4, stride=8
    ).fit(X)
    with pytest.raises(ValueError, match="horizon of 4"):
        model.predict(X, n=8)
    with pytest.raises(ValueError, match="context_length"):
        model.predict(_dataset(n_ts=4, sz=8, d=1))
    with pytest.raises(ValueError, match="features"):
        model.predict(_dataset(n_ts=4, sz=48, d=3))


def test_linear_probe_forecaster_fit_predict():
    X = _dataset(n_ts=6, sz=48, d=1)
    model = LinearProbeForecaster(
        _DummyBackbone(), context_length=16, horizon=2, stride=8
    )
    np.testing.assert_allclose(model.fit_predict(X, n=2), model.predict(X, n=2))


# ---------------------------------------------------------------------------
# LinearProbeClassifier
# ---------------------------------------------------------------------------


def _classification_dataset(n_per_class=15, sz=32, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 4 * np.pi, sz)
    sines = np.sin(t)[None, :] + 0.1 * rng.randn(n_per_class, sz)
    lines = np.linspace(-1, 1, sz)[None, :] + 0.1 * rng.randn(n_per_class, sz)
    X = np.concatenate([sines, lines])[:, :, None]
    y = np.array(["sine"] * n_per_class + ["line"] * n_per_class)
    return X, y


def test_linear_probe_classifier():
    X, y = _classification_dataset()
    model = LinearProbeClassifier(_DummyBackbone()).fit(X, y)

    assert set(model.classes_) == {"sine", "line"}
    assert model.predict(X).shape == (len(y),)
    assert model.predict_proba(X).shape == (len(y), 2)
    assert model.decision_function(X).shape == (len(y),)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0)
    # These two classes are easily separable
    assert model.score(X, y) > 0.9


def test_linear_probe_classifier_multivariate():
    X, y = _classification_dataset()
    X = np.concatenate([X, X[::-1]], axis=2)
    model = LinearProbeClassifier(_DummyBackbone()).fit(X, y)
    assert model.embedder_.embedding_size_ == 2 * D_MODEL
    assert model.predict(X).shape == (len(y),)


def test_linear_probe_classifier_accepts_any_probe():
    X, y = _classification_dataset()
    model = LinearProbeClassifier(_DummyBackbone(), probe=LinearSVC()).fit(X, y)
    assert isinstance(model.probe_, LinearSVC)
    with pytest.raises(AttributeError, match="predict_proba"):
        model.predict_proba(X)
    assert model.decision_function(X).shape == (len(y),)


def test_linear_probe_classifier_no_decision_function():
    X, y = _classification_dataset()
    model = LinearProbeClassifier(_DummyBackbone(), probe=GaussianNB()).fit(X, y)
    assert isinstance(model.probe_, GaussianNB)
    with pytest.raises(AttributeError, match="decision_function"):
        model.decision_function(X)
    assert model.predict_proba(X).shape == (len(y), 2)


def test_linear_probe_classifier_layer_and_pooling():
    X, y = _classification_dataset()
    for layer in (0, 1, None):
        for pooling in ("mean", "max", "token"):
            model = LinearProbeClassifier(
                _DummyBackbone(), layer=layer, pooling=pooling
            ).fit(X, y)
            assert model.predict(X).shape == (len(y),)


def test_linear_probe_classifier_transform():
    X, y = _classification_dataset()
    model = LinearProbeClassifier(_DummyBackbone()).fit(X, y)
    assert model.transform(X).shape == (len(y), D_MODEL)


def test_linear_probe_classifier_errors():
    X, y = _classification_dataset()
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        LinearProbeClassifier(_DummyBackbone()).fit(X, y[:-1])
    with pytest.raises(ValueError, match="input_layout"):
        LinearProbeClassifier(_DummyBackbone(), input_layout="bogus").fit(X, y)

    model = LinearProbeClassifier(_DummyBackbone()).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.concatenate([X, X], axis=2))


def test_estimators_are_clonable():
    for estimator in (
        TimeSeriesFoundationEmbedder(_DummyBackbone()),
        ZeroShotForecaster(_DummyPipeline()),
        LinearProbeForecaster(_DummyBackbone()),
        LinearProbeClassifier(_DummyBackbone()),
    ):
        cloned = clone(estimator)
        assert cloned is not estimator
        assert cloned.get_params().keys() == estimator.get_params().keys()
