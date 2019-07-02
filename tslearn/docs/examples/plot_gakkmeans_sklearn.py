from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_sklearn_dataset

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import numpy as np

# In this example, we show how tslearn estimators can be used in combination
# with sklearn utilities such as Pipeline
pipeline = Pipeline([
    ('normalize', TimeSeriesScalerMinMax()),
    ('clustering', GlobalAlignmentKernelKMeans())
])
n, sz, d = 15, 10, 3
rng = np.random.RandomState(0)
time_series = rng.randn(n, sz, d) * 10
print(time_series.shape)
pipeline.fit(time_series)
