# -*- coding: utf-8 -*-
"""
Early Classification
====================

This example presents the concept of early classification.

Early classifiers are implemented in the 
:mod:`tslearn.early_classification` module and in this example 
we use the method from [1]_.

References
----------

.. [1] A. Dachraoui, A. Bondu & A. Cornuejols. Early classification of time
  series as a non myopic sequential decision making problem. ECML/PKDD 2015
"""

# Author: Romain Tavenard
# License: BSD 3 clause
# sphinx_gallery_thumbnail_number = 2

from contextlib import suppress

import numpy

import matplotlib.animation as animation
import matplotlib.gridspec as gridpsec
import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.early_classification import NonMyopicEarlyClassifier
from tslearn.datasets import UCR_UEA_datasets

def plot_partial(time_series, t, y_true=0, y_pred=0, color="k"):
    plt.plot(time_series[:t+1].ravel(), color=color, linewidth=1.5)
    plt.plot(numpy.arange(t+1, time_series.shape[0]),
             time_series[t+1:].ravel(),
             linestyle="dashed", color=color, linewidth=1.5)
    plt.axvline(x=t, color=color, linewidth=1.5)
    plt.text(x=t - 20, y=time_series.max() - .25, s="Prediction time")
    plt.title(
        "Sample of class {} predicted as class {}".format(y_true, y_pred)
    )
    plt.xlim(0, time_series.shape[0] - 1)

##############################################################################
# Data loading and visualization
# ------------------------------

numpy.random.seed(0)
loader = UCR_UEA_datasets()
# sphinx_gallery_start_ignore
if "__file__" not in locals():
    # runs by sphinx-gallery
    import os
    loader._data_dir = os.path.join(
        os.path.dirname(os.path.realpath(os.getcwd())), '..', "datasets"
    )
# sphinx_gallery_end_ignore
X_train, y_train, X_test, y_test = loader.load_dataset("ECG200")

# Scale time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_test = TimeSeriesScalerMeanVariance().fit_transform(X_test)

size = X_train.shape[1]
n_classes = len(set(y_train))

plt.figure(layout="constrained")
for i, cl in enumerate(set(y_train)):
    ax = plt.subplot(n_classes, 1, i + 1)
    ax.set_title(f"Class {cl}")
    for ts in X_train[y_train == cl]:
        plt.plot(ts.ravel(), color="orange" if cl > 0 else "blue", alpha=.3)
    plt.xlim(0, size - 1)
plt.suptitle("Training time series")
plt.show()

##############################################################################
# Model fitting
# -------------
#
# As observed in the following figure, the optimal classification time as
# estimated by `NonMyopicEarlyClassifier` is data-dependent.

early_clf = NonMyopicEarlyClassifier(n_clusters=3,
                                     cost_time_parameter=1e-3,
                                     lamb=1e2,
                                     random_state=0)
early_clf = early_clf.fit(X_train, y_train)

preds, times = early_clf.predict_class_and_earliness(X_test)

plt.figure()
plt.subplot(2, 1, 1)
ts_idx = 0
t = times[ts_idx]
plot_partial(X_test[ts_idx], t, y_test[ts_idx], preds[ts_idx], color="orange")


plt.subplot(2, 1, 2)
ts_idx = 9
t = times[ts_idx]
plot_partial(X_test[ts_idx], t, y_test[ts_idx], preds[ts_idx], color="blue")
plt.tight_layout()
plt.show()

##############################################################################
# Streaming inputs
# ----------------
# Analysing early classification of a time series over time.

ts_index = 1
sz = X_test.shape[1]

fig = plt.figure(layout="constrained", figsize=(13, 4))
fig.suptitle(r"Optimal prediction time $\tau$ evolution")

gs = gridpsec.GridSpec(2, 3, figure=fig, width_ratios=[0.15, 0.70, 0.15])
ax1 = fig.add_subplot(gs[:, 0], title='Cluster probas')
ax2 = fig.add_subplot(
    gs[0, 1],
    xlim=[0, sz],
    ylim=[numpy.min(X_test[ts_index]),
          numpy.max(X_test[ts_index]) * 1.1],
    title='Streamed TS'
)
ax3 = fig.add_subplot(gs[1, 1], xlim=[0, sz], ylim=[0, 1], title='Expected cost')
ax4 = fig.add_subplot(gs[:, 2], title='Predicted probas')

bar1 = ax1.barh(
    ["cluster 1", "cluster 2", "cluster 3"],
    [1.1, 0, 0],
)
line1 = ax2.plot([numpy.nan], marker='.')[0]
line2 = ax3.plot(numpy.full((sz,), numpy.nan), linestyle="--",  marker='.')[0]
bar2 = ax4.bar(
    ["class -1", "class 1"],
    [1.1, 0],
)

def update(frame):
    incoming_ts_ =  X_test[ts_index, :frame+1]
    cluster_probas = early_clf.get_cluster_probas(incoming_ts_)
    expected_costs = early_clf._expected_costs(incoming_ts_).reshape(-1)
    probas, delays = early_clf.early_predict_proba(
        numpy.expand_dims(incoming_ts_, axis=0)
    )
    proba, delay = probas[0], delays[0]

    for i, elem in enumerate(bar1):
        elem.set_width(cluster_probas[i])

    line1.set_xdata(numpy.arange(incoming_ts_.shape[0]))
    line1.set_ydata(incoming_ts_)

    for i, elem in enumerate(bar2):
        elem.set_height(proba[i])

    with suppress(IndexError):
        ax2.lines[1].remove()
        ax3.lines[1].remove()
        ax2.texts[0].remove()
    ax2.axvline(x=frame + delay, color="k", linewidth=1.5)
    ax3.axvline(x=frame + delay, color="k", linewidth=1.5)
    ax2.text(x=frame + delay, y= numpy.max(X_test[ts_index])/2, s=r"$\tau$")
    line2.set_xdata(numpy.arange(expected_costs.shape[0]) + frame)
    line2.set_ydata(expected_costs)
    return bar1, line1, line2, bar2

ani = animation.FuncAnimation(fig=fig, func=update, frames=sz, interval=100)
plt.show()

##############################################################################
# Earliness-Accuracy trade-off
# ----------------------------
#
# The trade-off between earliness and accuracy is controlled via
# ``cost_time_parameter``.

plt.figure()
hatches = ["///", "\\\\\\", "*"]
for i, cost_t in enumerate([1e-4, 1e-3, 1e-2]):
    early_clf.set_params(cost_time_parameter=cost_t)
    early_clf.fit(X_train, y_train)
    preds, times = early_clf.predict_class_and_earliness(X_test)
    plt.hist(times,
             alpha=.5, hatch=hatches[i],
             density=True,
             label="$\\alpha={}$".format(cost_t),
             bins=numpy.arange(0, size, 5))
plt.legend(loc="upper right")
plt.xlim(0, size - 1)
plt.xlabel("Prediction times")
plt.title("Impact of cost_time_parameter ($\\alpha$)")
plt.show()
