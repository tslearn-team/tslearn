from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    ext_modules=cythonize(["tslearn/cydtw.pyx", "tslearn/cylrdtw.pyx"]),
    include_dirs=[numpy.get_include()],
    install_requires=['Cython', 'numpy', 'scipy', 'scikit-learn'],
    version="0.0.1",
    url="https://github.com/rtavenar/tslearn",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)