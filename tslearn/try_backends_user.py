import numpy as np
import torch
from try_backends_tslearn_functions import add, exp, log

x_numpy = np.array([1.0, 2.0, 0.1])
y_numpy = np.array([2.0, 3.0, -4.0])

x_torch = torch.tensor([1.0, 2.0, 3.0])
y_torch = np.array([1.0, 2.0, -10.0])

print("add")
print(add(x_numpy, y_numpy))
print(add(x_torch, y_torch))

print("exp")
print(exp(x_numpy))
print(exp(x_torch))

print("log")
print(log(x_numpy))
print(log(x_torch))
