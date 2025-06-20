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
            "License :: OSI Approved :: BSD License"
    ],
    packages=find_packages(),
    package_data={"tslearn": [".cached_datasets/singleTrainTest.csv", ".cached_datasets/Trace.npz"]},
    install_requires=[
        "scikit-learn<1.7",
        "numpy",
        "scipy",
        "numba",
        "joblib>=0.12",
        "h5py",
        "tensorflow==2.9.0; python_version == '3.8'",
        "tensorflow>=2; python_version != '3.8'"
    ],
    extras_require={'tests': ["pytest",
                              "matplotlib",
                              "cesium >= 0.12.2; python_version >= '3.9' and 'darwin' not in sys_platform",
                              "cesium; python_version < '3.9' or 'darwin' in sys_platform",
                              "pandas"],
                    'pytorch': ['torch']},
    version=get_version(),
    url="http://tslearn.readthedocs.io/",
    project_urls={
        "Source": "https://github.com/tslearn-team/tslearn",
    },
    author="Romain Tavenard",
    author_email="romain.tavenard@univ-rennes2.fr",
)
