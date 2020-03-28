import os

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
__version__ = "0.3.0"
__bibtex__ = r"""@misc{tslearn,
 title={tslearn: A machine learning toolkit dedicated to time-series data},
 author={Romain Tavenard and Johann Faouzi and Gilles Vandewiele and Felix Divo
         and Guillaume Androz and Chester Holtz and Marie Payne and
         Roman Yurchak and Marc Ru{\ss}wurm and Kushal Kolar and Eli Woods},
 year={2017},
 note={\url{https://github.com/rtavenar/tslearn}}
}"""


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    import pyximport
    pyximport.install()
