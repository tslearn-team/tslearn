"""
Linear probing a pre-trained model for classification
=====================================================

Time series foundation models are pre-trained for forecasting, yet the
representations they build along the way carry a lot of information about the
shape of a series, which makes them useful for other tasks as well. Linear
probing [1]_ is a cheap way to implement that principle: the pre-trained model 
is kept frozen and used as a feature extractor, and a plain classifier is fitted
on top of its representations. Because the head is linear and the backbone is
never updated, the accuracy reached tells us how linearly separable the classes
already are in the representation space.

This example applies :class:`~tslearn.foundation.LinearProbeClassifier` to a
UCR dataset, using Chronos-2 [2]_ as the frozen backbone. 
Running it requires the ``chronos-forecasting`` package::

    pip install "chronos-forecasting>=2.0"

References
----------
.. [1] G. Alain and Y. Bengio. Understanding intermediate layers using linear
  classifier probes. ICLR Workshop, 2017.
.. [2] A. F. Ansari, O. Shchur, J. Küken, et al. Chronos-2: From Univariate to
  Universal Forecasting. arXiv:2510.15821, 2025.
"""

# Author: Romain Tavenard
# License: BSD 3 clause
# sphinx_gallery_thumbnail_number = 2

##############################################################################
# Data
# ----
#
# We use the ``Trace`` dataset from the UCR archive, which gathers four classes
# of transient signals recorded in a nuclear power plant, with only 100 training
# series in total.

import numpy as np

from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

scaler = TimeSeriesScalerMeanVariance()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"{X_train.shape=}, {X_test.shape=}, {len(np.unique(y_train))} classes")

##############################################################################
# Probing the pre-trained model
# -----------------------------
#
# The classifier only needs the pre-trained model and, since Chronos-2 returns
# forecasts rather than hidden states, an explicit layer to read
# representations from. A forward hook is placed on the selected block, so no
# modification of the model is needed.
#
# Fitting is fast: each series goes through the model exactly once, and the
# only thing actually trained is a logistic regression over a few hundred
# features.

from chronos import Chronos2Pipeline

from tslearn.foundation import LinearProbeClassifier

pipeline = Chronos2Pipeline.from_pretrained("autogluon/chronos-2-small")

clf = LinearProbeClassifier(
    pipeline.model,
    layer=-2,
    pooling="mean",
    # Chronos-2 appends a register token and a forecast token after its context
    # tokens; neither represents the series, so they are left out of the average
    tokens=(0, -2),
)
clf.fit(X_train, y_train)

print(f"Embedding size: {clf.embedder_.embedding_size_}")
print(f"Test accuracy: {clf.score(X_test, y_test):.3f}")

##############################################################################
# Which layer, which pooling?
# ---------------------------
#
# The representations of a pre-trained model change a lot from one layer to the
# next, and the last ones are usually the most specialized towards the
# pre-training objective, here forecasting. Sweeping over layers is therefore
# worthwhile, and cheap: nothing is being trained beyond the linear head.
#
# ``pooling`` matters just as much. Chronos-2 emits one token per patch of the
# context plus a register token, so we compare averaging those tokens against
# picking the register token alone, the latter playing the role a ``[CLS]``
# token plays in a text encoder.

n_layers = len(pipeline.model.encoder.block)
poolings = ["mean", "max", "token"]
# Every other layer is enough to see the trend, and halves the build time
layers = sorted({*range(0, n_layers, 2), n_layers - 1})

accuracies = {}
for layer in layers:
    for pooling in poolings:
        model = LinearProbeClassifier(
            pipeline.model,
            layer=layer,
            pooling=pooling,
            # Chronos-2 places its register token after the context tokens
            token_index=-2 if pooling == "token" else 0,
            tokens=(0, -2),
        ).fit(X_train, y_train)
        accuracies[layer, pooling] = model.score(X_test, y_test)

header = "layer  " + "  ".join(f"{pooling:>6}" for pooling in poolings)
print(header)
for layer in layers:
    row = "  ".join(f"{accuracies[layer, pooling]:>6.3f}" for pooling in poolings)
    print(f"{layer:>5}  {row}")

##############################################################################

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
for pooling in poolings:
    ax.plot(
        layers,
        [accuracies[layer, pooling] for layer in layers],
        marker="o",
        label=f"pooling={pooling}",
    )
ax.set_xlabel("probed layer")
ax.set_ylabel("test accuracy")
ax.set_title("Linear probe accuracy across the layers of Chronos-2")
ax.legend()
plt.show()

##############################################################################
# Using the representations elsewhere
# -----------------------------------
#
# The feature extractor can also be used on its own, through
# :class:`~tslearn.foundation.TimeSeriesFoundationEmbedder`. Being a regular
# scikit-learn transformer, it composes with the rest of the ecosystem: here we
# project the frozen representations of the test set onto two dimensions to see
# whether the classes separate.

from sklearn.decomposition import PCA

from tslearn.foundation import TimeSeriesFoundationEmbedder

best_layer, best_pooling = max(accuracies, key=accuracies.get)
embedder = TimeSeriesFoundationEmbedder(
    pipeline.model, layer=best_layer, tokens=(0, -2)
)
embeddings = embedder.fit_transform(X_test)
projected = PCA(n_components=2).fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
for label in np.unique(y_test):
    mask = y_test == label
    ax.scatter(projected[mask, 0], projected[mask, 1], label=f"class {label}", s=20)
ax.set_xlabel("first principal component")
ax.set_ylabel("second principal component")
ax.set_title("Frozen representations of the Trace test set")
ax.legend()
plt.show()

##############################################################################
# Representations as time series
# ------------------------------
#
# Setting ``pooling=None`` skips the aggregation step altogether. Instead of
# one vector per series, the transformer then returns one vector per token,
# that is, a time series dataset of shape ``(n_ts, n_tokens, dim)``. Combined
# with ``tokens``, which drops the tokens that do not represent the input, this
# turns the pre-trained model into a time series to time series transform whose
# output can be fed to any other tslearn estimator.
#
# Chronos-2 emits one token per patch of 16 timesteps, so the representations
# below are a 16-times subsampled, high-dimensional view of the input series.
# Clustering those sequences under DTW recovers the classes reasonably well,
# without the labels ever being used.

from tslearn.clustering import TimeSeriesKMeans

from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score

embedder = TimeSeriesFoundationEmbedder(
    pipeline.model, layer=best_layer, pooling=None, tokens=(0, -2)
)
clusterer = TimeSeriesKMeans(
    n_clusters=len(np.unique(y_test)), metric="dtw", max_iter=5, random_state=0
)
pipeline = Pipeline(steps=[
    ("embed", embedder), 
    ("cluster", clusterer)
])
labels = pipeline.fit_predict(X_test)

print(f"Adjusted Rand index against the true classes: "
      f"{adjusted_rand_score(y_test, labels):.3f}")
