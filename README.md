Welcome to `tslearn`
====================

 |docs| 

`tslearn` is a Python package that provides machine learning tools for the analysis of time series.
This package builds on `scikit-learn`, `numpy` and `scipy` libraries.
At some point, it should be available on PyPI (as soon as it proves sufficiently helpful for the community).

Dependencies
------------

```
Cython
numpy
scipy
scikit-learn
```

Installation
------------

Run the following command for Cython code to compile:
```bash
python setup.py build_ext --inplace
```

Also, for the whole package to run properly, its base directory should be appended to your Python path.


Already available
-----------------

* A `generators` module provides Random Walks generators
* A `preprocessing` module provides standard time series scalers (implemented as `TransformerMixin`)
* A `metrics` module provides:
  * efficient Dynamic Time Warping (DTW) computation (derived from `cydtw` code)
  * efficient Locally-Regularized DTW (detailed presentation to come)
* A domain adaptation for time series module named `adaptation` that contains:
  * a method for (LR-)DTW-based non linear resampling that was previously released in `dtw_resample` repo
    * **Warning**: LR-DTW variant not tested yet!
* A `neighbors` module includes nearest neighbor algorithms to be used with time series
* A few examples are provided to serve as a doc while waiting for a proper one

TODO list
---------

* Add (Triangular) Global Alignment Kernel and soft-DTW to the proposed metrics
* Implement Learning Shapelets from Grabocka et al. (Conv+L2, + unsupervised)
* Add local feature extractors (`TransformerMixin`)
* Add k-means DBA by Petitjean _et al._ and soft-DTW k-means by Cuturi and Blondel
* Add metric learning for time series (Garreau _et al._)
* Add automatic retrieval of UCR/UEA datasets and 1M remote sensing time series
* Add LB_Keogh for nearest neighbor search
* Provide extensive documentation

.. |docs| image:: https://readthedocs.org/projects/docs/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://tslearn.readthedocs.io/en/latest/?badge=latest
