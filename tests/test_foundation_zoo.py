"""Tests exercising :mod:`tslearn.foundation` against real pre-trained models.

Unlike :mod:`test_foundation`, these tests download actual checkpoints from
the Hugging Face Hub and require model-specific packages (``timesfm``,
``uni2ts``, ``granite-tsfm``, ``momentfm``, a recent ``transformers``...)
that are *not* part of tslearn's own dependencies and that, in some cases,
pin conflicting versions of ``torch`` or ``transformers`` against one
another. Each test is therefore independent, skipped unless both its own
package is importable and the ``TSLEARN_RUN_FOUNDATION_ZOO`` environment
variable is set, and meant to be run in its own dedicated environment (see
``.github/workflows/test_foundation_zoo.yml``) rather than as part of the
regular test suite.

The code below mirrors the snippets given in the
``plot_foundation_model_zoo.py`` gallery example: this file is what backs
the claim, made there, that each recipe actually works.
"""

import os

import numpy as np

import pytest

from tslearn.generators import random_walks

pytestmark = pytest.mark.skipif(
    not os.environ.get("TSLEARN_RUN_FOUNDATION_ZOO"),
    reason="Set TSLEARN_RUN_FOUNDATION_ZOO=1 to run tests that download "
    "real pre-trained models from the Hugging Face Hub.",
)
torch = pytest.importorskip("torch", reason="torch not installed")

N_TS, SZ, D, CONTEXT_LENGTH, HORIZON = 5, 200, 1, 64, 12


@pytest.mark.parametrize("data", [
    random_walks(n_ts=N_TS, sz=SZ, random_state=0).astype(np.float64),
    torch.rand(N_TS, SZ, D, dtype=torch.float64),
])
def test_chronos_bolt(data):
    chronos = pytest.importorskip("chronos")

    from tslearn.foundation import LinearProbeForecaster, ZeroShotForecaster

    pipeline = chronos.BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small", device_map="cpu"
    )

    zero_shot = ZeroShotForecaster(pipeline)
    y_zero_shot = zero_shot.predict(data, n=HORIZON)
    assert y_zero_shot.shape == (N_TS, HORIZON, 1)

    probe = LinearProbeForecaster(
        pipeline.model,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        stride=8,
        layer=-1,
        layers_path="encoder.block",
        pooling="mean",
    )
    probe.fit(data)
    y_probe = probe.predict(data)
    assert y_probe.shape == (N_TS, HORIZON, 1)


@pytest.mark.parametrize("data", [
    random_walks(n_ts=N_TS, sz=SZ, random_state=0),
    torch.rand(N_TS, SZ, D, dtype=torch.float64),
])
def test_timesfm(data):
    timesfm = pytest.importorskip("timesfm")

    from tslearn.foundation import ZeroShotForecaster

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch"
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_LENGTH, max_horizon=HORIZON, normalize_inputs=True
        )
    )

    zero_shot = ZeroShotForecaster(
        model,
        predict_fn=lambda model, context, horizon: model.forecast(
            horizon=horizon, inputs=list(context)
        )[0],
        context_length=CONTEXT_LENGTH,
    )
    y_zero_shot = zero_shot.predict(data, n=HORIZON)
    assert y_zero_shot.shape == (N_TS, HORIZON, 1)


@pytest.mark.parametrize("data", [
    random_walks(n_ts=N_TS, sz=SZ, random_state=0),
    torch.rand(N_TS, SZ, D, dtype=torch.float64),
])
def test_moirai(data):
    pytest.importorskip("uni2ts")
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    from tslearn.foundation import ZeroShotForecaster

    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
    forecast_model = MoiraiForecast(
        module=module,
        prediction_length=HORIZON,
        context_length=CONTEXT_LENGTH,
        patch_size=32,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    def predict_fn(model, context, horizon):
        past_target = context.unsqueeze(-1)
        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros(past_target.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            return model(past_target, past_observed, past_is_pad, num_samples=20)

    zero_shot = ZeroShotForecaster(
        forecast_model, predict_fn=predict_fn, context_length=CONTEXT_LENGTH
    )
    y_zero_shot = zero_shot.predict(data, n=HORIZON)
    assert y_zero_shot.shape == (N_TS, HORIZON, 1)


@pytest.mark.parametrize("data", [
    random_walks(n_ts=N_TS, sz=SZ, random_state=0),
    torch.rand(N_TS, SZ, D, dtype=torch.float64),
])
def test_ttm(data):
    pytest.importorskip("tsfm_public")
    from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

    from tslearn.foundation import ZeroShotForecaster

    model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2", num_input_channels=1, revision="main"
    )

    def predict_fn(model, context, horizon):
        past_values = context.unsqueeze(-1)
        with torch.no_grad():
            return model(past_values=past_values).prediction_outputs

    zero_shot = ZeroShotForecaster(
        model, predict_fn=predict_fn, context_length=model.config.context_length
    )
    X = random_walks(
        n_ts=N_TS, sz=model.config.context_length + 50, random_state=0
    ).astype(np.float32)
    y_zero_shot = zero_shot.predict(X, n=model.config.prediction_length)
    assert y_zero_shot.shape == (N_TS, model.config.prediction_length, 1)


@pytest.mark.parametrize("data", [
    random_walks(n_ts=N_TS, sz=SZ, random_state=0),
    torch.rand(N_TS, SZ, D, dtype=torch.float64),
])
def test_moment(data):
    pytest.importorskip("momentfm")
    from momentfm import MOMENTPipeline

    from tslearn.foundation import LinearProbeForecaster

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"}
    )
    model.init()

    probe = LinearProbeForecaster(
        model,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        stride=32,
        layer=-1,
        layers_path="encoder.block",
        pooling="mean",
        input_layout="channels_first",
    )
    probe.fit(data)
    y_probe = probe.predict(data)
    assert y_probe.shape == (N_TS, HORIZON, 1)


@pytest.mark.parametrize("data", [
    random_walks(n_ts=N_TS, sz=SZ, random_state=0),
    torch.rand(N_TS, SZ, D, dtype=torch.float64),
])
def test_time_moe(data):
    transformers = pytest.importorskip("transformers")

    from tslearn.foundation import LinearProbeForecaster, ZeroShotForecaster

    model = transformers.AutoModelForCausalLM.from_pretrained(
        "Maple728/TimeMoE-50M", trust_remote_code=True
    )

    def predict_fn(model, context, horizon):
        out = model.generate(input_ids=context, max_new_tokens=horizon)
        return out[:, -horizon:]

    zero_shot = ZeroShotForecaster(model, predict_fn=predict_fn)
    y_zero_shot = zero_shot.predict(data, n=HORIZON)
    assert y_zero_shot.shape == (N_TS, HORIZON, 1)

    probe = LinearProbeForecaster(
        model,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        stride=8,
        layer=-1,
        layers_path="model.layers",
        pooling="last",
    )
    probe.fit(data)
    y_probe = probe.predict(data)
    assert y_probe.shape == (N_TS, HORIZON, 1)
