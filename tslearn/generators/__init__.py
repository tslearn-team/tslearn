"""
The :mod:`tslearn.generators` module gathers synthetic time series dataset
generation routines.
"""

from .generators import random_walks, random_walk_blobs

__all__ = ["random_walks", "random_walk_blobs"]
