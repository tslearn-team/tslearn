"""
The :mod:`tslearn.backend` module provides multiple backends.
The backends provided are NumPy and PyTorch.
"""

from .backend import Backend, backend_to_string, select_backend
from .base_backend import BaseBackend
from .numpy_backend import NumPyBackend
from .pytorch_backend import PyTorchBackend

__all__ = [
    "Backend",
    "backend_to_string",
    "select_backend",
    "BaseBackend",
    "NumPyBackend",
    "PyTorchBackend",
]
