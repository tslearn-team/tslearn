from setuptools import setup, find_packages
from codecs import open
import numpy
import os

# thanks Pipy for handling markdown now
ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()

# We can actually import a restricted version of tslearn that
# does not need the compiled code
import tslearn

VERSION = tslearn.__version__

# The following block
# is copied from sklearn as a way to avoid cythonizing source
# files in source dist
# We need to import setuptools before because it monkey-patches distutils
from distutils.command.sdist import sdist
cmdclass = {'sdist': sdist}

setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    long_description=README,
    long_description_content_type='text/markdown',
    license='BSD-2-Clause',
    classifiers=[
            "License :: OSI Approved :: BSD License"
    ],
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    package_data={"tslearn": [".cached_datasets/Trace.npz"]},
    install_requires=['numpy', 'scipy', 'scikit-learn', 'numba', 'joblib'],
    extras_require={'tests': ['pytest', 'torch'], 'pytorch': ['torch']},
    version=VERSION,
    url="http://tslearn.readthedocs.io/",
    project_urls={
        "Source": "https://github.com/tslearn-team/tslearn",
    },
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr",
    cmdclass=cmdclass
)
