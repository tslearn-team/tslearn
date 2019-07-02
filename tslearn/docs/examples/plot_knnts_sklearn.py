from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.generators import random_walk_blobs

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import numpy as np

# In this example, we show how tslearn estimators can be used in combination
# with sklearn utilities such as Pipeline
N_SPLITS = 3
pipeline = GridSearchCV(Pipeline([
    ('normalize', TimeSeriesScalerMinMax()),
    ('knn', KNeighborsTimeSeriesClassifier())
]), {'knn__n_neighbors': [3, 5, 7]}, cv=N_SPLITS, iid=True)

X, y = random_walk_blobs(n_ts_per_blob=25, sz=15, d=2, noise_level=2., 
                         random_state=42)
X = X * 10
pipeline.fit(X, y)
results = pipeline.cv_results_

print('Fitted KNeighborsTimeSeriesClassifier on random walk blobs...')
for i in range(len(results['params'])):
	s = '{}\t'.format(results['params'][i])
	for k in range(N_SPLITS):
		s += '{}\t'.format(results['split{}_test_score'.format(k)][i])
	print(s.strip())
