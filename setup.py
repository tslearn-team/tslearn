from setuptools import setup, Extension
import numpy
import re

from Cython.Distutils import build_ext as _build_ext

# dirty but working (from POT)
__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open('tslearn/__init__.py').read()).group(1)
# The beautiful part is, I don't even need to check exceptions here.
# If something messes up, let the build process fail noisy, BEFORE my release!

# thanks Pipy for handling markdown now
ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

list_pyx = ['cygak', 'cysax', 'cycc', 'soft_dtw_fast']
ext = [Extension('tslearn.%s' % s, ['tslearn/%s.pyx' % s],
                 include_dirs=[numpy.get_include()])
       for s in list_pyx]

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    long_description=README,
    long_description_content_type='text/markdown',
    include_dirs=[numpy.get_include()],
    packages=['tslearn'],
    package_data={"tslearn": [".cached_datasets/Trace.npz"]},
    data_files=[("", ["LICENSE"])],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'Cython', 'numba'],
    ext_modules=ext,
    cmdclass={'build_ext': _build_ext},
    version=__version__,
    url="http://tslearn.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)
