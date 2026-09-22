__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
__version__ = "0.10.0.dev"
__bibtex__ = r"""@article{JMLR:v21:20-091,
  author  = {Romain Tavenard and Johann Faouzi and Gilles Vandewiele and
             Felix Divo and Guillaume Androz and Chester Holtz and
             Marie Payne and Roman Yurchak and Marc Ru{\ss}wurm and
             Kushal Kolar and Eli Woods},
  title   = {Tslearn, A Machine Learning Toolkit for Time Series Data},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {118},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v21/20-091.html}
}"""

import importlib as _importlib

__all__ = {
    "backend",
    "barycenters",
    "bases",
    "clustering",
    "datasets",
    "early_classification",
    "forecasting",
    "foundation",
    "generators",
    "hdftools",
    "matrix_profile",
    "metrics",
    "neighbors",
    "neural_network",
    "piecewise",
    "preprocessing",
    "shapelets",
    "svm",
    "utils"
}


def __dir__():
    return [*globals().keys(), *__all__]


def __getattr__(name):
    if name in __all__:
        return _importlib.import_module(f"tslearn.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'tslearn' has no attribute '{name}'")
