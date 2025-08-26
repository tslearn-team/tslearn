"""
The :mod:`tslearn.shapelets` module gathers Shapelet-based algorithms.

It depends on the `keras` library (Keras3+ is required) and requires `pytorch`
as its computational backend. Be aware that keras backend must be configured before
importing Keras, and the backend cannot be changed after the package has been imported.
`tslearn` internally tries to set Keras backend, but cannot account for prior imports made
in a given execution context.

**User guide:** See the :ref:`Shapelets <shapelets>` section for further
details.
"""

from .shapelets import (
    LearningShapelets,
    grabocka_params_to_shapelet_size_dict
)

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

__all__ = [
    "LearningShapelets",
    "grabocka_params_to_shapelet_size_dict"
]
