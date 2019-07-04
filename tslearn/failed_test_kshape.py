from sklearn.base import clone
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.utils.testing import create_memmap_backed_data, set_random_state
from sklearn.metrics import adjusted_rand_score

from tslearn.clustering import KShape

import numpy as np


clusterer = KShape()
X, y = make_blobs(n_samples=50, random_state=1)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)
rng = np.random.RandomState(7)
X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])

#import matplotlib.pyplot as plt
#plt.figure()
#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

n_samples, n_features = X.shape
# catch deprecation and neighbors warnings
if hasattr(clusterer, "n_clusters"):
    clusterer.set_params(n_clusters=3)
set_random_state(clusterer)

# fit
clusterer.fit(X)
print(clusterer.cluster_centers_)

pred = clusterer.labels_
assert pred.shape == (n_samples,)

print(list(zip(pred, y)))

assert adjusted_rand_score(pred, y) > 0.4


"""
Traceback (most recent call last):
  File "tslearn/test_estimators.py", line 87, in <module>
    test_all_estimators()
  File "tslearn/test_estimators.py", line 78, in test_all_estimators
    estimators = get_estimators('all')
  File "tslearn/test_estimators.py", line 51, in get_estimators
    module = __import__(name, fromlist="dummy")
  File "/usr/local/lib/python3.6/dist-packages/tslearn/test_estimators.py", line 87, in <module>
    test_all_estimators()
  File "/usr/local/lib/python3.6/dist-packages/tslearn/test_estimators.py", line 84, in test_all_estimators
    check_estimator(estimator[1])
  File "/usr/local/lib/python3.6/dist-packages/sklearn/utils/estimator_checks.py", line 304, in check_estimator
    check(name, estimator)
  File "/usr/local/lib/python3.6/dist-packages/sklearn/utils/testing.py", line 350, in wrapper
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/sklearn/utils/estimator_checks.py", line 1299, in check_clustering
    assert_greater(adjusted_rand_score(pred, y), 0.4)
  File "/usr/lib/python3.6/unittest/case.py", line 1221, in assertGreater
    self.fail(self._formatMessage(msg, standardMsg))
  File "/usr/lib/python3.6/unittest/case.py", line 670, in fail
    raise self.failureException(msg)
AssertionError: 0.0 not greater than 0.4
"""
