"""
VARIMA
======

This example showcases the usage of the VARIMA and AutoVARIMA estimators for forecasting time series as described
in [1]_.

References
----------
.. [1] R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice. OTexts, 2014.
  https://otexts.com/fpp3/arima.html
"""

##############################################################################
# Retrieve data
# -------------
#
# This example uses open meteorological data from the Ille-et-Vilaine department
# from https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes.

import csv
import gzip
import urllib.request

import numpy as np

from tslearn.utils import to_time_series_dataset

def get_files():
    # sphinx_gallery_start_ignore
    if "__file__" not in locals():
        import os
        # runs by sphinx-gallery
        train_filename = os.path.join(
            os.path.dirname(os.path.realpath(os.getcwd())),
            "..",
            "datasets",
            "Q_35_previous-1950-2024_RR-T-Vent.csv.gz",
        )
        test_filename = os.path.join(
            os.path.dirname(os.path.realpath(os.getcwd())),
            "..",
            "datasets",
            "Q_35_latest-2025-2026_RR-T-Vent.csv.gz",
        )
        return train_filename, test_filename

    # sphinx_gallery_end_ignore
    train_filename, _ = urllib.request.urlretrieve(
        "https://www.data.gouv.fr/api/1/datasets/r/8051582c-2f1b-41c6-85a5-2f7f98389193"
    )
    test_filename, _ = urllib.request.urlretrieve(
        "https://www.data.gouv.fr/api/1/datasets/r/42760f87-0231-49b7-ac02-182ccced1f05"
    )
    return train_filename, test_filename

train_filename, test_filename = get_files()

def get_data(file_name):

    def format_date(date):
        return np.datetime64(date[0:4] + "-" + date[4:6] + "-" + date[6:8])

    data, times, places = [], [], []
    with gzip.open(file_name, "rt") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        next(reader)
        place = None
        for row in reader:
            if row[1] =="FOUGERES" or row[5] < '2010' or not row[8]:
                continue
            if row[1] != place:
                places.append(row[1])
                data.append([])
                times.append([])
                place = row[1]
            data[-1].append([float(row[8])])
            times[-1].append(format_date(row[5]))
    return to_time_series_dataset(data), times, places

test_data, test_times, test_places = get_data(test_filename)
train_data, train_times, train_places = get_data(train_filename)
indices = [train_places.index(city) for city in test_places]
train_data = train_data[indices]
train_times, train_places = [train_times[i] for i in indices], [train_places[i] for i in indices]


##############################################################################
# Models
# ------
#
# A VARIMA (Vectorized AutoRegressive Integrated Moving Average) model with hyperparameters :math:`p, d, q`,
# describes values at a given time :math:`t` based on both lagged values and lagged errors such that:
#
#  .. math::
#
#             y'_t = c + \phi_1 y'_{t-1} + ... + \phi_{p} y'_{t-p} +
#                        \theta_1 \epsilon_{t-1} + ... + \theta_{q} \epsilon_{t-q} +
#                        \epsilon_t
#
# where:
#  * :math:`y'` is the :math:`d` times differentiated timeseries.
#  * :math:`\epsilon_{t-k}` are lagged errors
#  * :math:`\epsilon_t` is white noise
#
# Fitting the model with data estimates the parameters :math:`c, \phi_1, ... \phi_p, \theta1, ..., \theta_q` through
# maximum likelihood estimation.
#
# Let's start toying with an arbitrary autoregressive model (p=2, d=0, q=0). In this model,
# forecasted values depends on the last two values. Hence, forecasting with a horizon :math:`n > 1`
# will use computed values for :math:`n-1, ..., 1`.

from sklearn.metrics import mean_absolute_error

from tslearn.forecasting import VARIMA

model = VARIMA(p=2, d=0, q=0).fit(train_data)

horizon = 7
predicted = model.predict(n=horizon)

MAE = np.mean([mean_absolute_error(test_data[i, :horizon], predicted[i]) for i in range(train_data.shape[0])])
print("MAE", MAE)

##############################################################################
#
# The AutoVARIMA model provides a way to automatically select the hyperparameters of the VARIMA model
# based on the training data. Selection of the order of differentiation :math:`d` aims at applying VARMA modeling onto
# stationary data whereas selection of :math:`p` and :math:`q` orders is driven by the minimization of AIC for
# relative VARMA models.

from tslearn.forecasting import AutoVARIMA

model = AutoVARIMA().fit(train_data)
print(f"Selected model: p={model.best_estimator_.p}, q={model.best_estimator_.q}, d={model.best_estimator_.d}")

predicted = model.predict(n=horizon)
MAE = np.mean([mean_absolute_error(test_data[i, :horizon], predicted[i]) for i in range(train_data.shape[0])])
print("MAE for selected model", MAE)

##############################################################################
#
# VARIMA and AutoARIMA estimators both allow for seasonally adjusted data.
# In that case, VARIMA modeling is performed against :math:`y_t - y_{t-k}` where :math:`k` is the seasonal period.

model = AutoVARIMA(seasonal_period=365).fit(train_data)

horizon = 365
predicted = model.predict(n=horizon)

import math
import matplotlib.pyplot as plt

plt.rcParams['date.converter'] = 'concise'
fig, axes = plt.subplots(math.ceil(len(train_data) / 3), 3, sharex=True, sharey=True,
                         layout="constrained", figsize = (13, 10))
for i in range(len(train_data)):
    j = i + 1
    ax = axes[j // 3][j % 3]
    predicted_plot, = ax.plot(test_times[0][:horizon], predicted[i], label="predicted")
    test_plot, = ax.plot(test_times[0][:horizon], test_data[i, :horizon], label="test", alpha=0.7)
    ax.set_title(test_places[i])
    for label in ax.get_xticklabels():
        label.set_rotation(40)
        label.set_horizontalalignment('right')
fig.delaxes(axes[0,0])
fig.legend(handles=[predicted_plot, test_plot], loc='upper left', bbox_to_anchor=(0.1, 0.95))
fig.suptitle("Forcasting seasonal data", fontsize=16)
plt.show()
