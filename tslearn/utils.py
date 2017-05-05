import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def npy2d_time_series(ts):
    if ts.ndim == 1:
        ts = ts.reshape((-1, 1))
    if ts.dtype != numpy.float:
        ts = ts.astype(numpy.float)
    return ts


def npy3d_time_series_dataset(dataset):
    if dataset.ndim == 2:
        dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], 1))
    if dataset.dtype != numpy.float:
        dataset = dataset.astype(numpy.float)
    return dataset


