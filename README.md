[![Documentation Status](https://readthedocs.org/projects/tslearn/badge/?version=latest)](http://tslearn.readthedocs.io/en/latest/?badge=latest)


`tslearn` is a Python package that provides machine learning tools for the analysis of time series.
This package builds on `scikit-learn`, `numpy` and `scipy` libraries.
At some point, it should be available on PyPI (as soon as it proves sufficiently helpful for the community).

# Dependencies

```
Cython
numpy
scipy
scikit-learn
```

# Installation

Run the following command for Cython code to compile:
```bash
python setup.py build_ext --inplace
```

Also, for the whole package to run properly, its base directory should be appended to your Python path.


# Already available

* A `generators` module provides Random Walks generators
* A `preprocessing` module provides standard time series scalers
* A `metrics` module provides:
  * Dynamic Time Warping (DTW) (derived from `cydtw` code)
  * Locally-Regularized DTW
  * Global Alignment Kernel
* A domain adaptation for time series module named `adaptation` contains:
  * a method for (LR-)DTW-based non linear resampling that was previously released in `dtw_resample` repo
* A `neighbors` module includes nearest neighbor algorithms to be used with time series
* A `clustering` module includes the following time series clustering algorithms:
  * Standard Euclidean k-means (based on `sklearn.cluster.KMeans` with adequate array reshaping done for you)
  * DBA k-means from Petitjean _et al._
  * Global Alignment kernel k-means

# TODO list

* Add soft-DTW to the proposed metrics
* Implement Learning Shapelets from Grabocka et al. (Conv+L2, + unsupervised)
* Add local feature extractors (`TransformerMixin`)
* Add soft-DTW k-means by Cuturi and Blondel
* Add metric learning for time series (Garreau _et al._)
* Add automatic retrieval of UCR/UEA datasets and 1M remote sensing time series
* Add LB_Keogh for nearest neighbor search
* Add Cost-Aware Early Classification of TS (Tavenard & Malinowski)?
