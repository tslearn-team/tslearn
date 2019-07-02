from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.generators import random_walk_blobs

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import numpy as np

# In this example, we show how tslearn estimators can be used in combination
# with sklearn utilities such as Pipeline
pipeline = GridSearchCV(Pipeline([
    ('normalize', TimeSeriesScalerMinMax()),
    ('knn', KNeighborsTimeSeriesClassifier())
]), {'knn__n_neighbors': [3, 5, 7]}, cv=2)

X, y = random_walk_blobs(n_ts_per_blob=50, sz=15, d=1)
X = X * 10
pipeline.fit(X, y)
print(pipeline.cv_results_)