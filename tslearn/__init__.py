import os

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'
__version__ = "0.1.20"
__bibtex__ = """@misc{tslearn,
 title={tslearn: A machine learning toolkit dedicated to time-series data},
 author={Tavenard, Romain},
 year={2017},
 note={\\url{https://github.com/rtavenar/tslearn}}
}"""


on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    import pyximport
    pyximport.install()
