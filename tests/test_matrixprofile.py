import numpy as np
import pytest

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def test_consistent_with_stumpy():
    pytest.importorskip('stumpy')
    import stumpy
    from tslearn.matrix_profile import MatrixProfile

    rng = np.random.RandomState(0)
    X = rng.randn(1, 20, 1)
    X_stumpy = X.ravel()

    mp = MatrixProfile(subsequence_length=10)
    mp_stumpy = MatrixProfile(subsequence_length=10, implementation="stump")

    X_tr = mp.fit_transform(X)
    X_tr_stumpy_wrap = mp_stumpy.fit_transform(X)
    X_tr_stumpy = stumpy.stump(X_stumpy, m=10)[:, 0].astype(float)

    np.testing.assert_allclose(X_tr.ravel(), X_tr_stumpy)
    np.testing.assert_allclose(X_tr, X_tr_stumpy_wrap)


@pytest.mark.parametrize("scale", [True, False])
def test_variable_length(scale):
    from tslearn.matrix_profile import MatrixProfile
    from tslearn.utils import to_time_series_dataset

    rng = np.random.RandomState(0)
    short_ts = rng.randn(15, 1)
    long_ts = rng.randn(20, 1)
    X = to_time_series_dataset([short_ts, long_ts])

    mp = MatrixProfile(subsequence_length=4, scale=scale)
    X_tr = mp.fit_transform(X)

    assert X_tr.shape == (2, 17, 1)
    np.testing.assert_allclose(X_tr[0, :12], mp.fit_transform([short_ts])[0])
    np.testing.assert_array_equal(X_tr[0, 12:], np.inf)
    np.testing.assert_allclose(X_tr[1], mp.fit_transform([long_ts])[0])
