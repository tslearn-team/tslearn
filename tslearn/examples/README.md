This `examples` folder contains the following notebooks :
* [`ex_metrics.ipynb`](ex_metrics.ipynb) refers to the following functions/classes:
  * `random_walks` from `tslearn.generators` module
  * `TimeSeriesScalerMeanVariance` from `tslearn.preprocessing` module
  * `dtw`, `dtw_path`, `lr_dtw_path` from `tslearn.metrics` module
* [`ex_neighbors.ipynb`](ex_neighbors.ipynb) refers to the following functions/classes:
  * `random_random_walk_blobs` from `tslearn.generators` module
  * `TimeSeriesScalerMinMax` from `tslearn.preprocessing` module
  * `KNeighborsDynamicTimeWarpingClassifier` from `tslearn.neighbors` module
* [`ex_adaptation.ipynb`](ex_adaptation.ipynb) refers to the following functions/classes:
  * `DTWSampler` from `tslearn.adaptation` module
  * `dtw` and `lr_dtw` from `tslearn.metrics` module (though not used directly, only through `DTWSampler`)
