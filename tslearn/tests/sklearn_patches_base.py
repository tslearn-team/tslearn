from contextlib import contextmanager

import numpy as np

from sklearn.utils._testing import ignore_warnings

from tslearn.generators import random_walk_blobs


def create_small_ts_dataset():
    return random_walk_blobs(n_ts_per_blob=5, n_blobs=3, random_state=1,
                             sz=10, noise_level=0.025)


def create_large_ts_dataset():
    return random_walk_blobs(n_ts_per_blob=50, n_blobs=3, random_state=1,
                             sz=20, noise_level=0.025)


@contextmanager
def patch(module, name, patched):
    orig = getattr(module, name)
    setattr(module, name, patched)
    try:
        yield
    finally:
        setattr(module, name, orig)


@ignore_warnings(category=FutureWarning)
def check_regressors_train(
    name, regressor_orig, readonly_memmap=False, X_dtype=np.float64
):
    import sklearn.utils.estimator_checks as checks
    with patch(checks, '_regression_dataset', create_large_ts_dataset):
        checks.check_regressors_train(name, regressor_orig, readonly_memmap, X_dtype)
