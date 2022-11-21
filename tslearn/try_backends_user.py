"""Functions used by the user calling the tslearn functions."""

import numpy as np
import torch
from backend.backend import Backend
from try_backends_tslearn_functions import add, exp, inv_matrices_main, log

x_numpy = np.array([1.0, 2.0, 0.1])
y_numpy = np.array([2.0, 3.0, -4.0])

x_torch = torch.tensor([1.0, 2.0, 3.0])
y_torch = np.array([1.0, 2.0, -10.0])

matrices_numpy = np.arange(8, dtype=float).reshape((2, 2, 2))
matrices_torch = torch.arange(8, dtype=float).reshape((2, 2, 2))

print("add")
print(add(x_numpy, y_numpy))
print(add(x_torch, y_torch))

print("exp")
print(exp(x_numpy))
print(exp(x_torch))

print("log")
print(log(x_numpy))
print(log(x_torch))

print("inv matrices")
print(inv_matrices_main(matrices_numpy))
print(inv_matrices_main(matrices_torch))

be = Backend()
b = be.array([[1, 2], [3, 4]])
print(b)

be = Backend("pytorch")
a = be.array([[1, 2], [3, 4]])
print(a)
