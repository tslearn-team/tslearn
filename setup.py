from setuptools import setup, Extension
import tslearn

from Cython.Distutils import build_ext as _build_ext


class CustomBuildExtCommand(_build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        _build_ext.run(self)

list_pyx = ['cydtw', 'cygak', 'cysax', 'cycc', 'soft_dtw_fast']
ext = [Extension('tslearn.%s' % s, ['tslearn/%s.pyx' % s]) for s in list_pyx]

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    packages=['tslearn'],
    package_data={"tslearn": [".cached_datasets/Trace.npz"]},
    data_files=[("", ["LICENSE"])],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'Cython'],
    ext_modules=ext,
    cmdclass={'build_ext': CustomBuildExtCommand},
    version=tslearn.__version__,
    url="http://tslearn.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)
