# Changelog
All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changelogs for this project are recorded in this file since v0.2.0.


## [Towards v0.8.0]

### Changed

* Fixed centroids computations in K-shape for multivariate timeseries ([#288](https://github.com/tslearn-team/tslearn/issues/288))

## [v0.7.0]

### Changed

* Explicit exception when using Global Alignment Kernel with sigma close to zero ([#440](https://github.com/tslearn-team/tslearn/issues/440))
* Fixed shifting in K-shape shape extraction process ([#385](https://github.com/tslearn-team/tslearn/issues/385))
* Support for `scikit-learn` up to 1.7 ([#549](https://github.com/tslearn-team/tslearn/issues/549))
* Fixed `LearningShapelets` with variable length timeseries ([#352](https://github.com/tslearn-team/tslearn/issues/352)) 
* The `shapelets` module now depends on Keras3+ and the underlying backend can be selected through
the KERAS_BACKEND environment variable. Defaults to the first found installed backend among `torch`,
`tensorflow` and `jax` (in that order).

### Removed

* Support for Python versions 3.8 and 3.9 is dropped

### Added

* `per_timeseries` and `per_feature` options for min-max and mean-variance scalers ([#536](https://github.com/tslearn-team/tslearn/issues/536))
* `TimeSeriesImputer`class: missing value imputer for time series ([#564](https://github.com/tslearn-team/tslearn/issues/564))
* Frechet metrics and KNeighbors integration ([#402](https://github.com/tslearn-team/tslearn/issues/402))

## [v0.6.4]

### Changed

* Support for `scikit-learn` up to 1.6

## [v0.6.3]

### Changed

* The structure of the class `Backend` has been simplified.

### Added

* Option `compute_with_backend` in functions `soft_dtw`, `soft_dtw_alignment`,
`cdist_soft_dtw` and `cdist_soft_dtw_normalized`.
`PyTorch` automatic differentiation can now be used when these functions are using the `PyTorch` backend.

### Fixed

* Fixed error in `LearningShapelets` when input parameter `n_shapelets_per_size` equals `None`.
* Fixed bug related to `SoftDTWLossPytorch` with option `normalize=True` when used on inputs of different lengths.
* Fixed error in function `from_hdf5` for array inputs.
* Fixed `readthedocs` test failing by replacing `build.image` (deprecated) with `build.os`.

## [v0.6.2]

### Fixed

* Fixed an incorrect calculation of the normalization term for `cdist_soft_dtw_normalized` when `dataset2` is provided.
* Fixed UCR/UEA datasets download link

## [v0.6.1]

### Fixed

* Fixed an import error when `torch` is not installed. This error appeared in tslearn v0.6.
`PyTorch` is now an optional dependency.

## [v0.6.0]

### Added

* Support of the `PyTorch` backend for the metrics of `tslearn`. 
In particular, the Dynamic Time Warping (DTW) metric and the Soft-DTW metric now support the `PyTorch` backend.

### Removed

* Support for Python version 3.7 is dropped
* Elements that were deprecated in v0.4 are now removed, as announced

## [v0.5.3]

### Changed

* Support for  `macOS-10.15` is replaced by support for `macOS-12`
* Support for `scikit-learn 0.23` is replaced by support for `scikit-learn 1.0`
* Specify supported `TensorFlow` version (2.9.0)

### Added

* Support for Python versions 3.9 and 3.10

### Fixed

* Fixed a bug about result of path in `lcss_path_from_metric` function
* Fixed incompatibilities between `NumPy`, `TensorFlow` and `scikit-learn` versions
* Fixed a bug preventing tslearn installation by removing the `NumPy` version constraint (<=1.19) in the file 
`pyproject.toml`

### Removed

* Cython is now replaced by Numba
* Support for Python versions 3.5 and 3.6 is dropped

## [v0.5.2]

### Changed

* In docs, change references to `master` branch to `main` branch.

## [v0.5.0]

### Changed

* Code refactoring to have all subpackages in subfolders
* Improved warnings in `datasets` loading
* `shapelets` module is now compatible with `tensorflow` 2.4

### Added

* Added canonical time warping (`ctw` and `ctw_path`)
* `soft_dtw_alignment` provides soft alignment path for soft-dtw
* `lcss` is a similarity measure based on the longest common subsequence
* `lcss_path_from_metric` allows one to pick a dedicated ground metric on top
of which the LCSS algorithm can be run

### Fixed

* numpy array hyper-parameters can now be serialized using `to_*()`
methods
* avoid `DivisionByZero` in `MinMaxScaler`
* Fixed incompatibilities with `scikit-learn` 0.24


## [v0.4.0]

### Changed

* k-means initialization function within `clustering/kmeans.py` updated
to be compatible with `scikit-learn` 0.24
* Better initialization schemes for `TimeSeriesKMeans` that lead to more
consistent clustering runs (helps avoid empty cluster situations)
* `TimeSeriesScalerMeanVariance` and `TimeSeriesScalerMinMax` are now
completely sklearn-compliant
* The `shapelets` module now requires tensorflow>=2 as dependency (was keras
tensorflow==1.* up to version 0.3)
* `GlobalAlignmentKernelKMeans` is deprecated in favor of `KernelKMeans` that
accepts various kernels (and "gak" is the default)
* `ShapeletModel` is now called `LearningShapelets` to be more explicit about
which shapelet-based classifier is implemented. `ShapeletModel` is still
available as an alias, but is now considered part of the private API

### Added

* Python 3.8 support
* `dtw_path_from_metric` allows one to pick a dedicated ground metric on top
of which the DTW algorithm can be run
* Nearest Neighbors on SAX representation (with custom distance)
* Calculate pairwise distance matrix between SAX representations
* `PiecewiseAggregateApproximation` can now handle variable lengths
* `ShapeletModel` is now serializable to JSON and pickle formats
* Multivariate datasets from the UCR/UEA archive are now available through
`UCR_UEA_datasets().load_dataset(...)`
* `ShapeletModel` now accepts variable-length time series dataset; a `max_size`
parameter has been introduced to save room at fit time for possibly longer
series to be fed to the model afterwards
* `ShapeletModel` now accepts a `scale` parameter that drives time series
pre-processing for better convergence
* `ShapeletModel` now has a public `history_` attribute that stores
loss and accuracy along fit epochs
* SAX and variants now accept a `scale` parameter that drives time series
pre-processing to fit the N(0,1) underlying hypothesis for SAX
* `TimeSeriesKMeans` now has a `transform` method that returns distances to
centroids
* A new `matrix_profile` module is added that allows `MatrixProfile` to be 
computed using the stumpy library or using a naive "numpy" implementation.
* A new `early_classification` module is added that offers early classification
estimators
* A new `neural_network` module is added that offers Multi Layer Perceptron
estimators for classification and regression

### Fixed

* Estimators that can operate on variable length time series now allow 
for test time datasets to have a different length from the one that was
passed at fit time
* Bugfix in `kneighbors()` methods.

### Removed

* Support for Python 2 is dropped


## [v0.3.1]

### Fixed

* Fixed a bug in `TimeSeriesSVC` and `TimeSeriesSVR` that caused user-input 
`gamma` to be ignored (always treated as if it were `"auto"`) for `gak` kernel


## [v0.3.0]

### Changed

* `dtw_barycenter_averaging` is made faster by using vectorized computations
* `dtw_barycenter_averaging` can be restarted several times to reach better
local optima using a parameter `n_init` set to 1 by default
* Functions `load_timeseries_txt` and `save_timeseries_txt` from the utils
module have changed their names to `load_time_series_txt` and 
`save_time_series_txt`. Old names can still be used but considered deprecated
and removed from the public API documentation for the sake of harmonization
* Default value for the maximum number of iterations to train `ShapeletModel` 
and `SerializableShapeletModel` is now set to 10,000 (used to be 100)
* `TimeSeriesScalerMeanVariance` and `TimeSeriesScalerMinMax` now ignore any
NaNs when calling their respective `transform` methods in order to better
mirror scikit-learn's handling of missing data in preprocessing.
* `KNeighborsTimeSeries` now accepts variable-length time series as inputs
when used with metrics that can deal with it (eg. DTW)
* When constrained DTW is used, if the name of the constraint is not given but 
its parameter is set, that is now considered sufficient to identify the 
constraint.

### Added

* `KNeighborsTimeSeriesRegressor` is a new regressor based on 
k-nearest-neighbors that accepts the same metrics as 
`KNeighborsTimeSeriesClassifier`
* A `set_weights` method is added to the `ShapeletModel` and  
`SerializableShapeletModel` estimators
* `subsequence_path` and `subsequence_cost_matrix` are now part of the public 
API and properly documented as such with an example use case in which more than
one path could be of interest (cf. `plot_sdtw.py`)
* `verbose` levels can be set for all functions / classes that use `joblib`
for parallel computations and `joblib` levels are used;
* conversion functions are provided in the `utils` module to interact with
other Python time series packages (`pyts`, `sktime`, `cesium`, `seglearn`, 
`tsfresh`, `stumpy`, `pyflux`)
* `dtw_barycenter_averaging_subgradient` is now available to compute DTW
barycenter based on subgradient descent
* `dtw_limited_warping_length` is provided as a way to compute DTW under upper
bound constraint on warping path length
* `BaseModelPackage` is a base class for serializing models to hdf5, json and 
pickle. h5py is added to requirements for hdf5 support.
* `BaseModelPackage` is used to add serialization functionality to the 
following models: `GlobalAlignmentKernelKMeans`, `TimeSeriesKMeans`,
`KShape`, `KNeighborsTimeSeries`, `KNeighborsTimeSeriesClassifier`,
`PiecewiseAggregateApproximation`, `SymbolicAggregateApproximation`,
and `OneD_SymbolicAggregateApproximation`

## [v0.2.4]

### Fixed

* The `tests` subdirectory is now made a python package and hence included in 
wheels

## [v0.2.2]

### Fixed

* The way version number is retrieved in `setup.py` was not working properly 
on Python 3.4 (and made the install script fail), switched back to the previous
version

## [v0.2.1]

### Added

* A `RuntimeWarning` is raised when an `'itakura'` constraint is set
that is unfeasible given the provided shapes.

### Fixed

* `'itakura'` and `'sakoe_chiba'` were swapped in `metrics.compute_mask`

## [v0.2.0]

### Added

* `tslearn` estimators are now automatically tested to match `sklearn`
requirements "as much as possible" (cf. `tslearn` needs in
terms of data format, _etc._)
* `cdist_dtw` and `cdist_gak` now have a `n_jobs` parameter to parallelize
distance computations using `joblib.Parallel`
* `n_jobs` is also available as a prameter in
`silhouette_score`, `TimeSeriesKMeans`, `KNeighborsTimeSeries`,
`KNeighborsTimeSeriesClassifier`, `TimeSeriesSVC`,
`TimeSeriesSVR` and `GlobalAlignmentKernelKMeans`

### Changed

* Faster DTW computations using `numba`
* `tslearn` estimators can be used in conjunction with `sklearn` pipelines and
cross-validation tools, even (for those concerned) with variable-length data
* doctests have been reduced to those necessary for documentation purposes, the
other tests being moved to `tests/*.py`
* The list of authors for the `tslearn` bibliographic reference has been
updated to include Johann Faouzi and Gilles Van de Wiele
* In `TimeSeriesScalerMinMax`, `min` and `max` parameters are now deprecated
in favor of `value_range`. Will be removed in v0.4
* In `TimeSeriesKMeans` and `silhouette_score`, `'gamma_sdtw'` is now
deprecated as a key for `metric_params` in favor of `gamma`. Will be removed
in v0.4

### Removed

* Barycenter methods implemented as estimators are no longer provided: use
dedicated functions from the `tslearn.barycenters` module instead
