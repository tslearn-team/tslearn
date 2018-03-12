from setuptools import setup, Extension
import numpy
import tslearn

from Cython.Distutils import build_ext as _build_ext

list_pyx = ['cydtw', 'cygak', 'cysax', 'cycc', 'soft_dtw_fast']
ext = [Extension('tslearn.%s' % s, ['tslearn/%s.pyx' % s], include_dirs=[numpy.get_include()]) for s in list_pyx]

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    include_dirs=[numpy.get_include()],
    packages=['tslearn'],
    package_data={"tslearn": [".cached_datasets/Trace.npz"]},
    data_files=[("", ["LICENSE"])],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'Cython'],
    ext_modules=ext,
    cmdclass={'build_ext': _build_ext},
    version=tslearn.__version__,
    url="http://tslearn.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)
