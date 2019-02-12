from setuptools import setup, Extension
import tslearn

# Inspired from a StackOverflow comment: https://stackoverflow.com/a/42163080

list_pyx = ['cydtw', 'cygak', 'cysax', 'cycc', 'soft_dtw_fast']
try:
    from Cython.setuptools import build_ext
except:
    # If we couldn't import Cython, use the normal setuptools
    # and look for a pre-compiled .c file instead of a .pyx file
    from setuptools.command.build_ext import build_ext
    ext_modules = [Extension('tslearn.%s' % s, ['tslearn/%s.c' % s])
                   for s in list_pyx]
else:
    # If we successfully imported Cython, look for a .pyx file
    ext_modules = [Extension('tslearn.%s' % s, ['tslearn/%s.pyx' % s])
                   for s in list_pyx]


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    packages=['tslearn'],
    package_data={"tslearn": [".cached_datasets/Trace.npz"]},
    data_files=[("", ["LICENSE"])],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'cython'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': CustomBuildExtCommand},
    version=tslearn.__version__,
    url="http://tslearn.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)
