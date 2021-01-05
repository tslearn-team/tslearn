"""
The :mod:`tslearn.early_classification` module gathers early classifiers for
time series.

Such classifiers aim at performing prediction as early as possible (i.e. they
do not necessarily wait for the end of the series before prediction is
triggered).

**User guide:** See the :ref:`Early Classification <early>` section for further 
 details.
"""

from .early_classification import NonMyopicEarlyClassifier

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

__all__ = [
    "NonMyopicEarlyClassifier"
]
