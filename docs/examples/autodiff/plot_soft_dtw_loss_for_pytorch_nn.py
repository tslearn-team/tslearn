# -*- coding: utf-8 -*-
"""
Soft-DTW loss for PyTorch neural network
========================================

This notebook is inspired by the notebook of Romain Tavenard about Alignment-based metrics in Machine Learning:
https://github.com/rtavenar/notebooks-ml4ts/blob/main/03_align4ml_sol.ipynb

The aim here is to use the Soft Dynamic Time Warping metric as a loss function for a PyTorch Neural Network.

We will rely on a `torch`-compatible implementation of soft-DTW obtained from github:
https://github.com/Maghoumi/pytorch-softdtw-cuda
"""

# Author: Yann Cabanes
# License: BSD 3 clause

# -*- coding: utf-8 -*-
"""Import the modules"""

import numpy as np
import matplotlib.pyplot as plt
import time
from tslearn.datasets import CachedDatasets
from tslearn.metrics import SoftDTWLossPyTorch
import torch
from torch import nn

"""Load the dataset

Using the CachedDatasets utility from tslearn, we load the "Trace" time series dataset. 
The dimensions of the arrays storing the time series training and testing datasets are (100, 275, 1).
We create a new dataset X_subset made of 50 random time series from classes indexed 1 to 3 
(y_train < 4) in the training set: X_subset is of shape (50, 275, 1).
"""

data_loader = CachedDatasets()
X_train, y_train, X_test, y_test = data_loader.load_dataset("Trace")

X_subset = X_train[y_train < 4]
np.random.shuffle(X_subset)
X_subset = X_subset[:50]

plt.title('Visualize a subset of the training dataset', size=20)
for ts in X_subset:
    plt.plot(ts[:, 0], color='k')

"""Multi-step ahead forecasting

In this section, our goal will be to implement a single-hidden-layer perceptron for time series forecasting. 
Our network will be trained to minimize normalized soft-DTW.
The normalized soft-DTW (also coined soft-DTW divergence) between the time series x and y is defined as: 
Soft-DTW(x, y) - (Soft-DTW(x, x) + Soft-DTW(y, y)) / 2
We will rely on a torch-compatible implementation of soft-DTW obtained from github:
https://github.com/Maghoumi/pytorch-softdtw-cuda
The code below is an implementation of a generic Multi-Layer-Perceptron class in torch, 
and we will rely on it for the implementation of a forecasting MLP with softDTW loss.
"""


class MultiLayerPerceptron(torch.nn.Module):  # No hidden layer here
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
            if e % 20 == 0:
                print('Epoch {}: train loss: {}'.format(e, loss.item()))
            # Backward pass
            loss.backward()
            self.optimizer.step()


"""Multilayer Perceptron model using the class above with PyTorch default loss function

We define an MLP class that would allow training a single-hidden-layer model using the 
default PyTorch loss function as a criterion to be optimized.
We train the network for 1000 epochs on a forecasting task that would consist, 
given the first 150 elements of a time series, in predicting the next 125 ones.
"""

model = MultiLayerPerceptron(
    layers=nn.Sequential(
        nn.Linear(in_features=150, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=125)
    )
)

print('\nMulti-step ahead forecasting using the PyTorch default loss function')
time_start = time.time()
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs=1000)  # Here one needs to define what X and y are, obviously
time_end = time.time()
print('Training time in seconds: ', time_end - time_start)

ts_index = 50

y_pred = model(X_test[:, :150, 0]).detach().numpy()

plt.figure()
plt.title('Multi-step ahead forecasting using\nthe PyTorch default loss function', size=20)
plt.plot(X_test[ts_index].ravel())
plt.plot(np.arange(150, 275), y_pred[ts_index], 'r-')


"""Multilayer perceptron using the Soft-DTW metric as a loss function

We take inspiration from the code above to define an MLP class that would allow training
a single-hidden-layer model using normalized soft-DTW as a criterion to be optimized.
We train the network for 100 epochs on a forecasting task that would consist, given the first 150 elements
of a time series, in predicting the next 125 ones.
"""

model = MultiLayerPerceptron(
    layers=nn.Sequential(
        nn.Linear(in_features=150, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=125)
    ),
    loss=SoftDTWLossPyTorch(gamma=0.1, normalize=False, dist_func=None)
)

print('\nMulti-step ahead forecasting using the Soft-DTW loss function')
time_start = time.time()
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs=100)
time_end = time.time()
print('Training time in seconds: ', time_end - time_start)

ts_index = 50

y_pred = model(X_test[:, :150, 0]).detach().numpy()

plt.figure()
plt.title('Multi-step ahead forecasting\nusing the Soft-DTW loss function', size=20)
plt.plot(X_test[ts_index].ravel())
plt.plot(np.arange(150, 275), y_pred[ts_index], 'r-')
plt.show()
