import os
from glob import glob
import numpy
import pytest
from sklearn.exceptions import NotFittedError
from tslearn import hdftools
from tslearn.metrics import sigma_gak
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

from tslearn.neighbors import KNeighborsTimeSeries, \
    KNeighborsTimeSeriesClassifier
from tslearn.clustering import KShape, TimeSeriesKMeans, \
    GlobalAlignmentKernelKMeans
from tslearn.generators import random_walks
from tslearn.piecewise import PiecewiseAggregateApproximation, \
    SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation


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


def _check_not_fitted(model):
    # not serializable if not fitted
    for fmt in all_formats:
        with pytest.raises(NotFittedError):
            getattr(model, "to_{}".format(fmt))(
                os.path.join(
                    tmp_dir, "{}.{}".format(model.__class__.__name__, fmt)
                )
            )


def _check_params_predict(model, X, test_methods):
    # serialize to all all_formats
    for fmt in all_formats:
        getattr(model, "to_{}".format(fmt))(
            os.path.join(
                tmp_dir, "{}.{}".format(model.__class__.__name__, fmt)
            )
        )

    # loaded models should have same model params
    # and provide the same predictions
    for fmt in all_formats:
        sm = getattr(model, "from_{}".format(fmt))(
            os.path.join(
                tmp_dir, "{}.{}".format(model.__class__.__name__, fmt)
            )
        )

        # make sure it's restored to the same class
        assert isinstance(sm, model.__class__)

        # test that predictions/transforms etc. are the same
        for method in test_methods:
            m1 = getattr(model, method)
            m2 = getattr(sm, method)
            numpy.testing.assert_equal(m1(X), m2(X))

        # check that the model-params are the same
        model_params = model._get_model_params()
        for p in model_params.keys():
            numpy.testing.assert_equal(getattr(model, p), getattr(sm, p))

        # check that hyper-params are the same
        hyper_params = model.get_params()
        for p in hyper_params .keys():
            numpy.testing.assert_equal(getattr(model, p), getattr(sm, p))

    clear_tmp()


def test_serialize_global_alignment_kernel_kmeans():
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")
    # Keep first 3 classes
    X_train = X_train[y_train < 4]
    numpy.random.shuffle(X_train)
    # Keep only 50 time series
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:50])
    sz = X_train.shape[1]

    gak_km = GlobalAlignmentKernelKMeans(n_clusters=3,
                                         sigma=sigma_gak(X_train),
                                         n_init=1,
                                         verbose=True,
                                         random_state=seed)

    _check_not_fitted(gak_km)

    gak_km.fit(X_train)

    _check_params_predict(gak_km, X_test, ['predict'])


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

    _check_not_fitted(dba_km)

    dba_km.fit(X_train)

    _check_params_predict(dba_km, X_train, ['predict'])

    sdtw_km = TimeSeriesKMeans(n_clusters=3,
                               metric="softdtw",
                               metric_params={"gamma_sdtw": .01},
                               verbose=True,
                               random_state=seed)

    _check_not_fitted(sdtw_km)

    sdtw_km.fit(X_train)

    _check_params_predict(sdtw_km, X_train, ['predict'])


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

    _check_not_fitted(ks)

    ks.fit(X_train)

    _check_params_predict(ks, X_train, ['predict'])


def test_serialize_knn():
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

    n_queries = 2
    n_neighbors = 4

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors)

    _check_not_fitted(knn)

    knn.fit(X_train)

    _check_params_predict(knn, X_test[:n_queries], ['kneighbors'])


def test_serialize_knn_classifier():
    seed = 0
    numpy.random.seed(seed)
    X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

    knc = KNeighborsTimeSeriesClassifier()

    _check_not_fitted(knc)

    knc.fit(X_train, y_train)

    _check_params_predict(knc, X_test, ['predict'])


def _get_random_walk():
    numpy.random.seed(0)
    # Generate a random walk time series
    n_ts, sz, d = 1, 100, 1
    dataset = random_walks(n_ts=n_ts, sz=sz, d=d)
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    return scaler.fit_transform(dataset)


def test_serialize_paa():
    X = _get_random_walk()
    # PAA transform (and inverse transform) of the data
    n_paa_segments = 10
    paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)

    _check_not_fitted(paa)

    paa.fit(X)

    _check_params_predict(paa, X, ['transform'])


def test_serialize_sax():
    n_paa_segments = 10
    n_sax_symbols = 8
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments,
                                         alphabet_size_avg=n_sax_symbols)

    _check_not_fitted(sax)

    X = _get_random_walk()

    sax.fit(X)

    _check_params_predict(sax, X, ['transform'])


def test_serialize_1dsax():

    n_paa_segments = 10
    n_sax_symbols_avg = 8
    n_sax_symbols_slope = 8

    one_d_sax = OneD_SymbolicAggregateApproximation(
        n_segments=n_paa_segments,
        alphabet_size_avg=n_sax_symbols_avg,
        alphabet_size_slope=n_sax_symbols_slope)

    _check_not_fitted(one_d_sax)

    X = _get_random_walk()
    one_d_sax.fit(X)

    _check_params_predict(one_d_sax, X, ['transform'])
