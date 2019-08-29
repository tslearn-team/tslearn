from setuptools import setup
from codecs import open
import numpy
import re
import os
from Cython.Build import cythonize

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
    extras_require={'tests': ['pytest']},
    ext_modules=cythonize("tslearn/*.pyx", include_path=[numpy.get_include()]),
    version=__version__,
    url="http://tslearn.readthedocs.io/",
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr"
)
