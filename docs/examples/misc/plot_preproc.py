"""
Preprocessing
=============
"""

import numpy as np
import matplotlib.pyplot as plt

#############################
# Imputer
# ---------------------------
# The :class:`.TimeSeriesImputer` class can be used to replace missing values (Nans)
# within a multi-variate variable-length dataset before feeding estimators.
#
# There are several available imputation methods (linear, mean, constant...)
# dealing with each dimension independently.
#
# .. note::
#
#     Tslearn pads with Nans in all dimensions when the dataset contains variable-length time series.
#     In that case, imputation should not affect the padding, hence the `keep_trailing_nans` parameter.
#     Use it to adjust the imputation behavior depending on your inputs.

from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesImputer

# Generate a random walk time series
n_ts, sz, d = 2, 100, 1
dataset = random_walks(n_ts=n_ts, sz=sz, d=d, random_state=42)

# Let's fake somme missing data / variable-length dataset
dataset[0, 25:35] = dataset[0, 70:80] = np.nan
dataset[1, 25:35] = dataset[1, 80:] = np.nan

imputed = TimeSeriesImputer(method="linear", keep_trailing_nans=True).fit_transform(dataset)

for i in range(n_ts):
    plt.plot(dataset[i], "b-", label="Input with missing data")
    plt.plot(
        imputed[i].reshape(-1),
        'b:',
        label="Linearly imputed data"
    )
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()


###########################
# Scalers
# -------------------------
# Feature scaling is an important preprocessing step for many machine learning algorithms.
# Because time series are two-dimensionnal, the scaling axis matters in both
# :class:`.TimeSeriesScalerMinMax` and :class:`.TimeSeriesScalerMeanVariance`.
#
# Scaling each time series of a dataset independently is a common practice that somehow keeps
# the shapes unchanged. In this case, no data is persisted between `fit` and `transform`.
#
# Nevertheless, when the magnitude of the signal is important, scaling can be performed dataset wise,
# and scaling parameters computed on the training set are persisted to scale the test data.

import calendar

import numpy as np

from tslearn.utils import to_time_series_dataset

dataset = to_time_series_dataset([
    [0, 0, 1, 3, 7, 10, 12, 11, 9, 6, 3, 1],
    [11, 14, 17, 20, 24, 25, 26, 25, 25, 21, 17 ,12],
    [7, 7 , 5 , 3 , 1, -1, -1, 0, 1, 3, 4, 6],
    [24, 24, 24, 23, 21, 20, 19, 19, 20, 21, 22, 23]
])
cities = [
    ("Groninguen", "N", "C"),
    ("Shenzen", "N", "H"),
    ("Punta Arenas", "S", "C"),
    ("Rio", "S", "H")
]

def visu(data, title):
    plt.figure(figsize=(12, 3))
    for i, x in enumerate(data):
        label, hemisphere, climate = cities[i]
        plt.plot(
            x,
            label=label,
            linestyle="--" if hemisphere == "N" else ":",
            color = "b" if climate == "C" else "r",
            alpha=0.6
        )
        plt.xticks(np.arange(0, 12), list(calendar.month_abbr)[1:])
        plt.ylabel("min T")
        plt.legend()
        plt.title(title)
    plt.show()

visu(dataset, "Data")

######################################
#
# Scaling per timeseries keeps the shapes of the time series at the expense of amplitude standardization
# as shown in the example below.

from tslearn.preprocessing import TimeSeriesScalerMinMax
scaled = TimeSeriesScalerMinMax(per_timeseries=True).fit_transform(dataset)
visu(scaled, "Data scaled per time series")

######################################
#
# When scaling is done dataset wise, the relative amplitudes are kept.
#

scaled = TimeSeriesScalerMinMax(per_timeseries=False).fit_transform(dataset)
visu(scaled, "Data scaled dataset wise")


#############################
# Feature synchronizer
# --------------------
# :class:`.TimeSeriesFeatureSynchronizer` is a preprocessing step dealing with
# features acquired at different sampling rates or desynchronized timestamps.

def plot_(dataset, timestamps=None, title=None, xlim=None):
    fig, axes = plt.subplots(len(dataset), 1, sharex=True, sharey=True, layout="tight")
    if title:
        fig.suptitle(title)
    axes = [axes] if len(dataset)< 2 else axes

    plots = [None] * dataset.shape[-1]
    for ts_index, ts in enumerate(dataset):
        for k in range(ts.shape[-1]):
            x = timestamps[ts_index, :, k] if timestamps is not None else np.arange(len(ts))
            plots[k], = axes[ts_index].plot(x, ts[..., k].reshape(-1), "o--" , alpha=0.7, label=f"feature {k}")
            axes[ts_index].set_title(f"TS {ts_index}")
            if xlim is not None:
                axes[ts_index].set_xlim(right=xlim)
            if timestamps is not None:
                for label in axes[ts_index].get_xticklabels():
                    label.set_rotation(40)
                    label.set_horizontalalignment('right')
    fig.legend(handles=plots)
    plt.show()

from tslearn.preprocessing import TimeSeriesFeatureSynchronizer

######################################
# For multi-variate variable-length datasets, the feature synchronization is performed
# such that all dimensions reach a common sampling rate.
#
points = np.cos(np.linspace(0, 5 * np.pi, 1000))
subsampled_points_0 = points[::10]
subsampled_points_1 = points[::13]
subsampled_points_2 = points[::20]
subsampled_points_3 = points[::27]

n_points = max(
    len(subsampled_points_0),
    len(subsampled_points_1),
    len(subsampled_points_2),
    len(subsampled_points_3)
)

dataset = np.full((2, n_points, 3), np.nan)
dataset[0, :len(subsampled_points_0),  0] = subsampled_points_0
dataset[0, :len(subsampled_points_1),  1] = subsampled_points_1
dataset[0, :len(subsampled_points_2),  2] = subsampled_points_2

dataset[1, :len(subsampled_points_3),  0] = subsampled_points_3
dataset[1, :len(subsampled_points_1),  1] = subsampled_points_1
dataset[1, :len(subsampled_points_2),  2] = subsampled_points_2

plot_(dataset, title="Dataset")

synchronized = TimeSeriesFeatureSynchronizer().fit_transform(dataset)
plot_(synchronized, title="Synchronized against feature 0", xlim=n_points)

synchronized = TimeSeriesFeatureSynchronizer(reference_feature_index=1).fit_transform(dataset)
plot_(synchronized, title="Synchronized against feature 1", xlim=n_points)

synchronized = TimeSeriesFeatureSynchronizer(reference_feature_index=2).fit_transform(dataset)
plot_(synchronized, title="Synchronized against feature 2", xlim=n_points)


######################################
# When timestamps are not provided, the target sampling rate is assumed to be constant.
# For finer tuning, timestamps can be injected into the synchronization to
# account for irregular acquisition.

rng = np.random.default_rng(seed=0)

days = np.arange('2005-02', '2005-04', dtype='datetime64[D]')
points = np.cos(np.linspace(0, 5 * np.pi, len(days)))

n_points = len(days)

timestamps = np.full((1, n_points, 3), np.datetime64("NaT"), dtype="datetime64[D]")
dataset = np.full((1, n_points, 3), np.nan)
indices_1 = np.sort(rng.choice(np.arange(n_points), int(n_points//2), replace=False))
indices_2 = np.sort(rng.choice(np.arange(n_points), int(n_points//1.5), replace=False))

timestamps[..., 0] = days
timestamps[:, :len(indices_1), 1] = days[indices_1]
timestamps[:, :len(indices_2), 2] = days[indices_2]
dataset[..., 0] = points
dataset[:, :len(indices_1), 1] = points[indices_1]
dataset[:, :len(indices_2), 2] = points[indices_2]

plot_(dataset, timestamps, "Dataset")

synchronized = TimeSeriesFeatureSynchronizer().fit_transform(dataset,
                                                             timestamps=timestamps)
plot_(synchronized,
      days.reshape(1, -1, 1).repeat(3, axis=2),
      "Synchronized against feature 0")

synchronized = TimeSeriesFeatureSynchronizer(reference_feature_index=1).fit_transform(dataset,
                                                                                      timestamps=timestamps)
plot_(synchronized,
      days[indices_1].reshape(1, -1, 1).repeat(3, axis=2),
      "Synchronized against feature 1",
      xlim=days[-1])

synchronized = TimeSeriesFeatureSynchronizer(reference_feature_index=2).fit_transform(dataset,
                                                                                      timestamps=timestamps)
plot_(
    synchronized,
    days[indices_2].reshape(1, -1, 1).repeat(3, axis=2),
    "Synchronized against feature 2",
    xlim=days[-1])
