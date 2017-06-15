from setuptools import setup, Extension
import numpy

have_cython = False
try:
    from Cython.Distutils import build_ext as _build_ext
    have_cython = True
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext

list_pyx = ['cydtw', 'cygak', 'cylrdtw', 'cysax', 'cycc']
if have_cython:
    ext = [Extension('tslearn.%s' % s, ['tslearn/%s.pyx' % s]) for s in list_pyx]
else:
    ext = [Extension('tslearn.%s' % s, ['tslearn/%s.c' % s]) for s in list_pyx]

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    include_dirs=[numpy.get_include()],
    packages=['tslearn'],
    install_requires=['numpy', 'scipy', 'scikit-learn'],
    ext_modules=ext,
    cmdclass={'build_ext': _build_ext},
    version="0.0.23",
    url="http://tslearn.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)