import os
from glob import glob
import numpy
from tslearn import hdftools
from tslearn.clustering import KShape
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')


os.makedirs(tmp_dir, exist_ok=True)


def teardown_module():
    files = glob(os.path.join(tmp_dir, '*'))
    for f in files:
        os.remove(f)
    os.removedirs(tmp_dir)


def test_hdftools():
    dtypes = [numpy.int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
              numpy.float, numpy.float32, numpy.float64, numpy.float128]

    d = {}

    for dtype in dtypes:
        name = numpy.dtype(dtype).name
        d[name] = (numpy.random.rand(100, 100) * 10).astype(dtype)

    fname = os.path.join(tmp_dir, 'hdf_test.hdf5')

    hdftools.save_dict(d, filename=fname, group='data')

    d2 = hdftools.load_dict(fname, 'data')

    for k in d2.keys():
        numpy.testing.assert_equal(d[k], d2[k])


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

    # Euclidean k-means
    ks = KShape(n_clusters=3, verbose=True, random_state=seed)
    y_pred = ks.fit_predict(X_train)

    formats = ['json', 'hdf5', 'pickle']

    for fmt in formats:

        getattr(ks, "to_{}".format(fmt))(
            os.path.join(tmp_dir, "ks.{}".format(fmt))
        )

    for fmt in formats:

        sm = getattr(ks, "from_{}".format(fmt))(
            os.path.join(tmp_dir, "ks.{}".format(fmt))
        )

        assert isinstance(sm, KShape)

        numpy.testing.assert_equal(y_pred, sm.predict(X_train))
        numpy.testing.assert_equal(ks.norms_centroids_, sm.norms_centroids_)
        numpy.testing.assert_equal(ks.norms_, sm.norms_)
        numpy.testing.assert_equal(ks.cluster_centers_, sm.cluster_centers_)


