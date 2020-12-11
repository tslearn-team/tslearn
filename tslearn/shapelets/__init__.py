"""
The :mod:`tslearn.shapelets` module gathers Shapelet-based algorithms.

It depends on the `tensorflow` library for optimization (TF2 is required).

**User guide:** See the :ref:`Shapelets <shapelets>` section for further 
details.
"""

from .shapelets import LearningShapelets, ShapeletModel, \
    SerializableShapeletModel, grabocka_params_to_shapelet_size_dict

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

__all__ = [
    "LearningShapelets", "ShapeletModel", "SerializableShapeletModel",
    "grabocka_params_to_shapelet_size_dict"
]