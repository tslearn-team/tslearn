# -*- coding: utf-8 -*-
"""
Soft-DTW loss for PyTorch neural network
========================================

The aim here is to use the Soft Dynamic Time Warping metric as a loss function of a PyTorch Neural Network for
time series forecasting.

The `torch`-compatible implementation of the soft-DTW loss function is available from the
:mod:`tslearn.metrics` module.
"""

# Authors: Yann Cabanes, Romain Tavenard
# License: BSD 3 clause
# sphinx_gallery_thumbnail_number = 2

"""Import the modules"""

import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import CachedDatasets
from tslearn.metrics import SoftDTWLossPyTorch
import torch
from torch import nn

##############################################################################
# Load the dataset
# ----------------
#
# Using the CachedDatasets utility from tslearn, we load the "Trace" time series dataset. 
# The dimensions of the arrays storing the time series training and testing datasets are (100, 275, 1).
# We create a new dataset X_subset made of 50 random time series from classes indexed 1 to 3 
# (y_train < 4) in the training set: X_subset is of shape (50, 275, 1).

data_loader = CachedDatasets()
X_train, y_train, X_test, y_test = data_loader.load_dataset("Trace")

X_subset = X_train[y_train < 4]
np.random.shuffle(X_subset)
X_subset = X_subset[:50]

##############################################################################
# Multi-step ahead forecasting
# ----------------------------
#
# In this section, our goal is to implement a single-hidden-layer perceptron for time series forecasting. 
# Our network will be trained to minimize the soft-DTW metric.
# We will rely on a `torch`-compatible implementation of the soft-DTW loss function.
# The code below is an implementation of a generic Multi-Layer-Perceptron class in torch, 
# and we will rely on it for the implementation of a forecasting MLP with softDTW loss.

# Note that Soft-DTW can take negative values due to the regularization parameter gamma.
# The normalized soft-DTW (also coined soft-DTW divergence) between the time series x and y is defined as: 
# Soft-DTW(x, y) - (Soft-DTW(x, x) + Soft-DTW(y, y)) / 2
# The normalized Soft-DTW is always positive.
# However, the computation time of the normalized soft-DTW equals three times the computation time of the Soft-DTW.

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, layers, loss=None):
        # At init, we define our layers
        super(MultiLayerPerceptron, self).__init__()
        self.layers = layers
        if loss is None:
            self.loss = torch.nn.MSELoss(reduction="none")
        else:
            self.loss = loss
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, X):
        # The forward method informs about the forward pass: how one computes outputs of the network
        # from the input and the parameters of the layers registered at init
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        batch_size = X.size(0)
        X_reshaped = torch.reshape(X, (batch_size, -1))  # Manipulations to deal with time series format
        output = self.layers(X_reshaped)
        return torch.reshape(output, (batch_size, -1, 1))  # Manipulations to deal with time series format

    def fit(self, X, y, max_epochs=10):
        # The fit method performs the actual optimization
        X_torch = torch.Tensor(X)
        y_torch = torch.Tensor(y)

        for e in range(max_epochs):
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.forward(X_torch)
            # Compute Loss
            loss = self.loss(y_pred, y_torch).mean()
            # Backward pass
            loss.backward()
            self.optimizer.step()


##############################################################################
# Using MSE as a loss function
# ----------------------------
# 
# We define an MLP class that would allow training a single-hidden-layer model using 
# mean squared error (MSE) as a loss function to be optimized.
# We train the network for 1000 epochs on a forecasting task that would consist, 
# given the first 150 elements of a time series, in predicting the next 125 ones.

model = MultiLayerPerceptron(
    layers=nn.Sequential(
        nn.Linear(in_features=150, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=125)
    )
)

# Here one needs to define what X and y are, obviously
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs=1000)

ts_index = 50
y_pred = model(X_test[:, :150, 0]).detach().numpy()

plt.figure()
plt.title('Multi-step ahead forecasting using MSE')
plt.plot(X_test[ts_index].ravel())
plt.plot(np.arange(150, 275), y_pred[ts_index], 'r-')


##############################################################################
# Using Soft-DTW as a loss function
# ---------------------------------
#
# We take inspiration from the code above to define an MLP class that would allow training
# a single-hidden-layer model using soft-DTW as a criterion to be optimized.
# We train the network for 100 epochs on a forecasting task that would consist, given the first 150 elements
# of a time series, in predicting the next 125 ones.

model = MultiLayerPerceptron(
    layers=nn.Sequential(
        nn.Linear(in_features=150, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=125)
    ),
    loss=SoftDTWLossPyTorch(gamma=0.1)
)

model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs=100)

y_pred = model(X_test[:, :150, 0]).detach().numpy()

plt.figure()
plt.title('Multi-step ahead forecasting using Soft-DTW loss')
plt.plot(X_test[ts_index].ravel())
plt.plot(np.arange(150, 275), y_pred[ts_index], 'r-')
