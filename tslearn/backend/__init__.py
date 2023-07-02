"""
The :mod:`tslearn.backend` module provides multiple backends.
The backends provided are NumPy and PyTorch.
"""

from .backend import Backend, instantiate_backend, select_backend
from .numpy_backend import NumPyBackend
from .pytorch_backend import PyTorchBackend

__all__ = [
    "Backend",
    "instantiate_backend",
    "select_backend",
    "NumPyBackend",
    "PyTorchBackend",
]
