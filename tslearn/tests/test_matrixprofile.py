import numpy as np
import pytest

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_consistent_woth_stumpy():
    pytest.importorskip('stumpy')
    import stumpy
    from tslearn.matrix_profile import MatrixProfile

    rng = np.random.RandomState(0)
    X = rng.randn(1, 20, 1)
    X_stumpy = X.ravel()

    mp = MatrixProfile(subsequence_length=10)
    X_tr = mp.fit_transform(X)
    X_tr_stumpy = stumpy.stump(X_stumpy, m=10)[:, 0].astype(np.float)

    np.testing.assert_allclose(X_tr.ravel(), X_tr_stumpy)