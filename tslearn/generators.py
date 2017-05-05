import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def random_walks(n_ts=100, sz=256, d=1, mu=0., std=1.0):
    ts = numpy.empty((n_ts, sz, d))
    rnd = numpy.random.randn(n_ts, sz, d) * std + mu
    ts[:, 0, :] = rnd[:, 0, :]
    for t in range(1, sz):
        ts[:, t, :] = ts[:, t - 1, :] + rnd[:, t, :]
    return ts


def random_walk_blobs(n_ts_per_blob=100, sz=256, d=1, n_blobs=2, noise_level=1.0):
    base_ts = random_walks(n_ts=n_blobs, sz=sz, d=d, std=1.0)
    rnd = numpy.random.randn(n_ts_per_blob * n_blobs, sz, d) * noise_level
    ts = numpy.repeat(base_ts, repeats=n_ts_per_blob, axis=0)
    y = numpy.repeat(range(n_blobs), repeats=n_ts_per_blob)
    return ts + rnd, y
