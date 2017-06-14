from setuptools import setup, Extension
import numpy

have_cython = False
try:
    from Cython.Distutils import build_ext as _build_ext
    have_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext

if have_cython:
    ext = [
        Extension('tslearn.cydtw', ['tslearn/cydtw.pyx']),
        Extension('tslearn.cygak', ['tslearn/cygak.pyx']),
        Extension('tslearn.cylrdtw', ['tslearn/cylrdtw.pyx']),
        Extension('tslearn.cysax', ['tslearn/cysax.pyx']),
        Extension('tslearn.cycc', ['tslearn/cycc.pyx'])
    ]
else:
    ext = [
        Extension('tslearn.cydtw', ['tslearn/cydtw.c']),
        Extension('tslearn.cygak', ['tslearn/cygak.c']),
        Extension('tslearn.cylrdtw', ['tslearn/cylrdtw.c']),
        Extension('tslearn.cysax', ['tslearn/cysax.c']),
        Extension('tslearn.cycc', ['tslearn/cycc.c'])
    ]

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    include_dirs=[numpy.get_include()],
    packages=['tslearn'],
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    ext_modules=ext,
    cmdclass={'build_ext': _build_ext},
    version="0.0.22",
    url="http://tslearn.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)