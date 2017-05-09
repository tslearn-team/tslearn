from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, RadiusNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsRegressor

from tslearn.metrics import dtw

class NearestNeighborsDynamicTimeWarping(NearestNeighbors):
    """Unsupervised learner for implementing neighbor searches using Dynamic Time Warping as the core metric."""
    def __init__(self, n_neighbors=5, radius=1.0, **kwargs):
        NearestNeighbors.__init__(self, n_neighbors=n_neighbors, radius=radius, 
                                  algorithm='brute', metric=dtw, **kwargs)

class KNeighborsDynamicTimeWarpingClassifier(KNeighborsClassifier):
    """Classifier implementing the k-nearest neighbors vote with Dynamic Time Warping as its core metric."""
    def __init__(self, n_neighbors=5, weights='uniform', **kwargs):
        KNeighborsClassifier.__init__(self, n_neighbors=n_neighbors, weights=weights, 
                                      algorithm='brute', metric=dtw, **kwargs)

class RadiusNeighborsDynamicTimeWarpingClassifier(RadiusNeighborsClassifier):
    """Classifier implementing the k-nearest neighbors vote with Dynamic Time Warping as its core metric."""
    def __init__(self, radius=5, weights='uniform', outlier_label=None, **kwargs):
        RadiusNeighborsClassifier.__init__(self, radius=radius, weights=weights, 
                                           outlier_label=outlier_label, algorithm='brute', 
                                           metric=dtw, **kwargs)

class KNeighborsDynamicTimeWarpingRegressor(KNeighborsRegressor):
    """Regression based on k-nearest neighbors with Dynamic Time Warping as the core metric."""
    def __init__(self, n_neighbors=5, weights='uniform', **kwargs):
        KNeighborsRegressor.__init__(self, n_neighbors=n_neighbors, weights=weights, 
                                     algorithm='brute', metric=dtw, **kwargs)

class RadiusNeighborsDynamicTimeWarpingRegressor(RadiusNeighborsRegressor):
    """Regression based on neighbors within a fixed radius with Dynamic Time Warping as the core metric."""
    def __init__(self, radius=5, weights='uniform', **kwargs):
        RadiusNeighborsRegressor.__init__(self, radius=radius, weights=weights, 
                                          algorithm='brute', metric=dtw, **kwargs)
