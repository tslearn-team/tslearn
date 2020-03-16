# Changelog
All notable changes to this project will be documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changelogs for this project are recorded in this file since v0.2.0.

## [Towards v0.4.0]

### Changed
* TimeSeriesScalerMeanVariance and TimeSeriesScalerMinMax are now completely sklearn-compliant
* Bugfix in kneighbors() methods.

### Added
* Nearest Neighbors on SAX representation (with custom distance)
* Calculate pairwise distance matrix between SAX representations
* PiecewiseAggregateApproximation can now handle variable lengths

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

### Changed
 
* When constrained DTW is used, if the name of the constraint is not given but 
its parameter is set, that is now considered sufficient to identify the 
constraint.

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
