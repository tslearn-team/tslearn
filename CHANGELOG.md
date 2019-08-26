# Changelog
All notable changes to this project will be documented in this file.

The format is based on 
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to 
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changelogs for this project are recorded in this file since v0.2.0.

## [v0.2.0]

### Added

* `tslearn` estimators are now automatically tested to match `sklearn` 
requirements "as much as possible" (eg. while still fitting `tslearn` needs in 
terms of data format, _etc._)

### Changed

* Faster DTW computations using `numba`
* `tslearn` estimators can be used in conjunction with `sklearn` pipelines and
cross-validation tools, even (for those concerned) with variable-length data
* doctests have been reduced to those necessary for documentation purposes, the
other tests being moved to `tests/*.py`
* The list of authors for the `tslearn` bibliographic reference has been 
updated to include Johann Faouzi and Gilles Van de Wiele

### Removed

* Barycenter methods implemented as estimators are no longer provided: use
dedicated functions from the `tslearn.barycenters` module instead
