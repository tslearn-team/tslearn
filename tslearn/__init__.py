import os

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
__version__ = "0.5.3.2"
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


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    import pyximport
    pyximport.install()
