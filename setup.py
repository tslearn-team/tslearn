from setuptools import setup, find_packages
from codecs import open
import os

ROOT = os.path.abspath(os.path.dirname(__file__))

def get_readme():
    with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
        readme = f.read()
    return readme

def get_version():
    with open(os.path.join(ROOT, "tslearn/__init__.py")) as fd:
        data = fd.readlines()
        version_line = next(line for line in data if line.startswith("__version__"))

    return version_line.strip().split("=")[1].strip('"\' ')


setup(
    name="tslearn",
    description="A machine learning toolkit dedicated to time-series data",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    license='BSD-2-Clause',
    classifiers=[
            "BSD-2-Clause"
    ],
    packages=find_packages(),
    package_data={"tslearn": [".cached_datasets/singleTrainTest.csv", ".cached_datasets/Trace.npz"]},
    python_requires='>=3.9',
    install_requires=[
        "scikit-learn>=1.4,<1.7",
        "numpy>=1.24.3,<2.3",
        "scipy>=1.10.1,<1.17",
        "numba>=0.58.1,<0.62",
        "joblib>=1.2,<1.6"
    ],
    extras_require={
        "pytorch": ['torch'],
        "tests": [
            "pytest",
            "torch",
            # Torch dropped support of macOS intel after version 2.2.2
            # which does not support numpy2.
            "numpy<2; 'darwin' in sys_platform and 'x86_64' in platform_machine",
            "h5py",
            "keras",
            "stumpy"
        ],
        "docs": [
            "sphinx",
            "sphinx_bootstrap_theme",
            "sphinx-gallery",
            "numpydoc",
            "nbsphinx",
            "matplotlib",
            "ipykernel",
            "pypandoc"
        ],
        "all_features": [
            "torch",
            # Torch dropped support of macOS intel after version 2.2.2
            # which does not support numpy2.
            "numpy<2; 'darwin' in sys_platform and 'x86_64' in platform_machine",
            "h5py",
            "keras",
            "cesium",
            "pandas"
        ]
    },
    version=get_version(),
    url="http://tslearn.readthedocs.io/",
    project_urls={
        "Source": "https://github.com/tslearn-team/tslearn",
    },
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr",
)
