# `tslearn`

`tslearn` is a Python package that provides machine learning tools for the analysis of time series.
This package builds on `scikit-learn`, `numpy` and `scipy` libraries.
At some point, it should be available on PyPI (as soon as it proves sufficiently helpful for the community).

## Already available

* A `metrics` module provides efficient DTW computation (derived from `cydtw` code)

## TODO list

* Add standard time-series scalers (`TransformerMixin`)
* Integrate `dtw_resample`
* Add (Triangular) Global Alignment Kernel, LR-DTW and soft-DTW to the proposed metrics
* Implement Learning Shapelets from Grabocka et al. (Conv+L2, + unsupervised)
* Add local feature extractors (`TransformerMixin`)
* Add k-means DBA by Petitjean _et al._ and soft-DTW k-means by Cuturi and Blondel
* Provide extensive documentation