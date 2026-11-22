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


# MatrixProfile per implementation: results on worked examples, how stumpy is
# called, NaN handling, the errors raised at transform time, and the params.
_TS = [0., 1., 3., 2., 9., 1., 14., 15., 1., 2., 2., 10., 7.]


def test_numpy_worked_example():
    from tslearn.matrix_profile import MatrixProfile

    mp = MatrixProfile(subsequence_length=4, scale=False)
    expected = np.sqrt([47., 2., 38., 63., 130., 184., 324., 195., 2., 38.])
    np.testing.assert_allclose(mp.fit_transform([_TS])[0, :, 0], expected)


def test_numpy_scaled_and_transform_after_fit():
    from tslearn.matrix_profile import MatrixProfile

    # current output on main, pinned before the refactoring
    expected = [0.642486, 0.285705, 1.640169, 0.898131, 1.279547,
                1.781965, 2.987226, 2.839433, 0.285705, 0.642486]
    mp = MatrixProfile(subsequence_length=4)
    np.testing.assert_allclose(mp.fit_transform([_TS])[0, :, 0], expected,
                               atol=1e-6)
    np.testing.assert_allclose(mp.fit([_TS]).transform([_TS])[0, :, 0],
                               expected, atol=1e-6)


def test_numpy_ignores_nan_subsequences():
    from tslearn.matrix_profile import MatrixProfile

    ts = [0., 1., 0., 1., np.nan, 0., 1., 0., 1.]
    mp = MatrixProfile(subsequence_length=2, scale=False)
    np.testing.assert_array_equal(mp.fit_transform([ts])[0, :, 0],
                                  [0., 0., 0., np.inf, np.inf, 0., 0., 0.])


def test_gpu_stump_calls_stumpy_per_series(monkeypatch):
    stumpy = pytest.importorskip('stumpy')
    from tslearn.matrix_profile import MatrixProfile

    calls = []

    def fake_gpu_stump(T_A, m):
        calls.append((T_A.copy(), m))
        n = len(T_A) - m + 1
        # like stumpy, an object array whose column 0 is the profile
        out = np.zeros((n, 4), dtype=object)
        out[:, 0] = np.arange(n) + 10 * len(calls)
        return out

    monkeypatch.setattr(stumpy, "gpu_stump", fake_gpu_stump)
    X = np.array([_TS[:8], _TS[5:]]).reshape((2, 8, 1))
    X_tr = MatrixProfile(subsequence_length=3,
                         implementation="gpu_stump").fit_transform(X)

    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], _TS[:8])
    np.testing.assert_array_equal(calls[1][0], _TS[5:])
    assert calls[0][1] == calls[1][1] == 3
    assert X_tr.dtype == float
    np.testing.assert_array_equal(X_tr[:, :, 0],
                                  [np.arange(6) + 10, np.arange(6) + 20])


@pytest.mark.parametrize("implementation", ["stump", "gpu_stump"])
def test_stumpy_implementations_need_stumpy(monkeypatch, implementation):
    from tslearn.matrix_profile import MatrixProfile
    from tslearn.matrix_profile import matrix_profile

    monkeypatch.setattr(matrix_profile, "STUMPY_INSTALLED", False)
    mp = MatrixProfile(subsequence_length=4, implementation=implementation)
    with pytest.raises(ImportError, match="stumpy is not installed"):
        mp.fit_transform([_TS])


def test_unknown_implementation_raises_at_transform():
    from tslearn.matrix_profile import MatrixProfile

    mp = MatrixProfile(subsequence_length=4, implementation="foo")
    mp.fit([_TS])
    msg = ('This "foo" matrix profile implementation is not recognized. '
           "Available implementations are ['numpy', 'stump', 'gpu_stump'].")
    with pytest.raises(ValueError) as excinfo:
        mp.transform([_TS])
    assert str(excinfo.value) == msg


@pytest.mark.parametrize("implementation", ["numpy", "stump", "foo"])
def test_multivariate_input_raises_first(implementation):
    from tslearn.matrix_profile import MatrixProfile

    mp = MatrixProfile(subsequence_length=2, implementation=implementation)
    with pytest.raises(NotImplementedError, match="multi-dimensional"):
        mp.fit_transform(np.zeros((1, 5, 2)))


def test_implementation_is_a_plain_param():
    from sklearn.base import clone
    from tslearn.matrix_profile import MatrixProfile

    mp = MatrixProfile(subsequence_length=4, scale=False,
                       implementation="stump")
    assert mp.get_params()["implementation"] == "stump"
    assert clone(mp).get_params()["implementation"] == "stump"

    mp.set_params(implementation="foo").fit([_TS])
    with pytest.raises(ValueError, match='"foo"'):
        mp.transform([_TS])
    mp.set_params(implementation="numpy")
    np.testing.assert_allclose(mp.transform([_TS])[0, 0, 0], np.sqrt(47.))
