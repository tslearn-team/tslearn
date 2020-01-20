import os
from glob import glob
import numpy
import pytest
from sklearn.exceptions import NotFittedError
from tslearn import hdftools
from tslearn.clustering import KShape, TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler


tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
all_formats = ['json', 'hdf5', 'pickle']

try:
    os.makedirs(tmp_dir)
except (FileExistsError, OSError):
    pass


def teardown_module():
    clear_tmp()
    os.removedirs(tmp_dir)


def clear_tmp():
    files = glob(os.path.join(tmp_dir, '*'))
    for f in files:
        os.remove(f)


def test_hdftools():
    dtypes = [numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
              numpy.float, numpy.float32, numpy.float64]

    d = {}

    for dtype in dtypes:
        name = numpy.dtype(dtype).name
        d[name] = (numpy.random.rand(100, 100) * 10).astype(dtype)

    fname = os.path.join(tmp_dir, 'hdf_test.hdf5')

    hdftools.save_dict(d, filename=fname, group='data')

    d2 = hdftools.load_dict(fname, 'data')

    for k in d2.keys():
        numpy.testing.assert_equal(d[k], d2[k])


def _not_fitted_check(model):
    # not serializable if not fitted
    with pytest.raises(NotFittedError):
        for fmt in all_formats:
            getattr(model, "to_{}".format(fmt))(
                os.path.join(tmp_dir, "ks.{}".format(fmt))
            )


def _params_predict_check(model, X):
    # serialize to all all_formats
    for fmt in all_formats:
        getattr(model, "to_{}".format(fmt))(
            os.path.join(tmp_dir, "ks.{}".format(fmt))
        )

    # loaded models should have same model params
    # and provide the same predictions
    for fmt in all_formats:
        sm = getattr(model, "from_{}".format(fmt))(
            os.path.join(tmp_dir, "ks.{}".format(fmt))
        )

        # make sure it's restored to the same class
        assert isinstance(sm, model.__class__)

        # test that saved model gives same predictions
        numpy.testing.assert_equal(model.predict(X), sm.predict(X))

        # check that the model-params are the same
        model_params = model._get_model_params()
        for p in model_params.keys():
            numpy.testing.assert_equal(getattr(model, p), getattr(sm, p))

        # check that hyper-params are the same
        hyper_params = model.get_params()
        for p in hyper_params .keys():
            numpy.testing.assert_equal(getattr(model, p), getattr(sm, p))

    clear_tmp()

def test_serialize_timeserieskmeans():
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    X_train = X_train[y_train < 4]  # Keep first 3 classes
    numpy.random.shuffle(X_train)
    # Keep only 50 time series
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
    # Make time series shorter
    X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
    sz = X_train.shape[1]

    dba_km = TimeSeriesKMeans(n_clusters=3,
                              n_init=2,
                              metric="dtw",
                              verbose=True,
                              max_iter_barycenter=10,
                              random_state=seed)

    _not_fitted_check(dba_km)

    dba_km.fit(X_train)

    _params_predict_check(dba_km, X_train)

    sdtw_km = TimeSeriesKMeans(n_clusters=3,
                               metric="softdtw",
                               metric_params={"gamma_sdtw": .01},
                               verbose=True,
                               random_state=seed)

    _not_fitted_check(sdtw_km)

    sdtw_km.fit(X_train)

    _params_predict_check(sdtw_km, X_train)


def test_serialize_kshape():
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    # Keep first 3 classes
    X_train = X_train[y_train < 4]
    numpy.random.shuffle(X_train)
    # Keep only 50 time series
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
    sz = X_train.shape[1]

    ks = KShape(n_clusters=3, verbose=True, random_state=seed)

    _not_fitted_check(ks)

    ks.fit(X_train)

    _params_predict_check(ks, X_train)




