# `tslearn`

`tslearn` is a Python package that provides machine learning tools for the analysis of time series.
This package builds on `scikit-learn`, `numpy` and `scipy` libraries.
At some point, it should be available on PyPI (as soon as it proves sufficiently helpful for the community).

## Dependencies

```
Cython
numpy
scipy
scikit-learn
```

## Installation

Run the following command for Cython code to compile:
```bash
python setup.py build_ext --inplace
```

Also, for the whole package to run properly, its base directory should be appended to your Python path.


## Already available

* A `generators` module provides Random Walks generators
* A `preprocessing` module provides standard time series scalers (implemented as `TransformerMixin`)
* A `metrics` module provides:
  * efficient DTW computation (derived from `cydtw` code)
  * efficient Locally-Regularized DTW (detailed presentation to come)
* A domain adaptation for time series module named `adaptation` that contains:
  * a method for (LR-)DTW-based non linear resampling that was previously released in `dtw_resample` repo
    * **Warning**: LR-DTW variant not tested yet!
* A few examples are provided to serve as a doc while we build a proper one

## TODO list

* Add (Triangular) Global Alignment Kernel and soft-DTW to the proposed metrics
* Implement Learning Shapelets from Grabocka et al. (Conv+L2, + unsupervised)
* Add local feature extractors (`TransformerMixin`)
* Add k-means DBA by Petitjean _et al._ and soft-DTW k-means by Cuturi and Blondel
* Add automatic retrieval of UCR/UEA datasets
* Provide extensive documentation