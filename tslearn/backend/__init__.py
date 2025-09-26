"""
The :mod:`tslearn.backend` module provides multiple backends.
The backends provided are NumPy and PyTorch.
"""
import os
import logging
from importlib import import_module

from .backend import Backend, cast, instantiate_backend, select_backend


def check_keras_backend():
    """
    Select a backend based on installed packages
    when none is explicitly selected.

    """
    if not os.environ.get("KERAS_BACKEND"):
        for keras_backend in ["torch", "tensorflow", "jax"]:
            try:
                import_module(keras_backend)
                os.environ["KERAS_BACKEND"] = keras_backend
                logging.info(
                    "Using %s as Keras backend, may be overloaded through " +
                    "the `KERAS_BACKEND` environment variable." % keras_backend
                )
                break
            except ModuleNotFoundError:
                logging.debug(
                    "Skipping %s backend for Keras: not installed" % keras_backend
                )
        else:
            raise ImportError("No Keras backend installed")


__all__ = [
    "Backend",
    "cast",
    "instantiate_backend",
    "select_backend",
    'check_keras_backend'
]
