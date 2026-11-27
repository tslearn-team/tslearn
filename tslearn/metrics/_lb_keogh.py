import numpy

from numba import njit, prange

from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


def lb_keogh(ts_query, ts_candidate=None, radius=1, envelope_candidate=None):
    r"""Compute LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts_query : array-like, shape=(sz1, 1) or (sz1,)
        Univariate query time series to compare to the envelope of the candidate.
    ts_candidate : None or array-like, shape=(sz2, 1) or (sz2,) (default: None)
        Univariate candidate time series. None means the envelope is provided via
        `envelope_candidate` parameter and hence does not
        need to be computed again.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on
        all observations from the candidate time series at indices comprised
        between i-radius and i+radius). Not used
        if `ts_candidate` is None.
    envelope_candidate: pair of array-like (envelope_down, envelope_up) or None
    (default: None)
        Pre-computed envelope of the candidate time series. If set to None, it
        is computed based on `ts_candidate`.

    Notes
    -----
        This method requires a `ts_query` and `ts_candidate` (or
        `envelope_candidate`, depending on the call) to be of equal size.

    Returns
    -------
    float
        Distance between the query time series and the envelope of the
        candidate time series.

    Examples
    --------
    >>> ts1 = [1, 2, 3, 2, 1]
    >>> ts2 = [0, 0, 0, 0, 0]
    >>> env_low, env_up = lb_envelope(ts1, radius=1)
    >>> float(lb_keogh(ts_query=ts2,
    ...                envelope_candidate=(env_low, env_up)))  # doctest: +ELLIPSIS
    2.8284...
    >>> float(lb_keogh(ts_query=ts2,
    ...                ts_candidate=ts1,
    ...                radius=1))  # doctest: +ELLIPSIS
    2.8284...

    See also
    --------
    lb_envelope : Compute LB_Keogh-related envelope

    References
    ----------
    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """
    if ts_candidate is None:
        envelope_down, envelope_up = envelope_candidate
    else:
        ts_candidate = to_time_series(ts_candidate)
        assert (
            ts_candidate.shape[1] == 1
        ), "LB_Keogh is available only for univariate time series"
        envelope_down, envelope_up = lb_envelope(ts_candidate, radius)
    ts_query = to_time_series(ts_query)
    assert (
        ts_query.shape[1] == 1
    ), "LB_Keogh is available only for univariate time series"
    indices_up = ts_query[:, 0] > envelope_up[:, 0]
    indices_down = ts_query[:, 0] < envelope_down[:, 0]
    return numpy.sqrt(
        numpy.linalg.norm(ts_query[indices_up, 0] - envelope_up[indices_up, 0]) ** 2
        + numpy.linalg.norm(ts_query[indices_down, 0] - envelope_down[indices_down, 0])
        ** 2
    )


@njit()
def _njit_lb_envelope(time_series, radius):
    """Compute time series envelope as required by LB_Keogh.

    Parameters
    ----------
    time_series : array-like, shape=(sz, d)
        Time series for which the envelope should be computed.
    radius : int
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on all observations from the time series
        at indices comprised between i-radius and i+radius).

    Returns
    -------
    envelope_down : array-like, shape=(sz, d)
        Lower-side of the envelope.
    envelope_up : array-like, shape=(sz, d)
        Upper-side of the envelope.
    """
    sz, d = time_series.shape
    envelope_up = numpy.empty((sz, d))
    envelope_down = numpy.empty((sz, d))

    for i in prange(sz):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        for di in prange(d):
            envelope_down[i, di] = numpy.min(time_series[min_idx:max_idx, di])
            envelope_up[i, di] = numpy.max(time_series[min_idx:max_idx, di])

    return envelope_down, envelope_up


def _lb_envelope(time_series, radius, be=None):
    """Compute time series envelope as required by LB_Keogh.

    Parameters
    ----------
    time_series : array-like, shape=(sz, d)
        Time series for which the envelope should be computed.
    radius : int
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on all observations from the time series
        at indices comprised between i-radius and i+radius).
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    envelope_down : array-like, shape=(sz, d)
        Lower-side of the envelope.
    envelope_up : array-like, shape=(sz, d)
        Upper-side of the envelope.
    """
    be = instantiate_backend(be, time_series)
    time_series = be.array(time_series)
    sz, d = be.shape(time_series)
    envelope_up = be.empty((sz, d))
    envelope_down = be.empty((sz, d))

    for i in range(sz):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        for di in range(d):
            envelope_down[i, di] = be.min(time_series[min_idx:max_idx, di])
            envelope_up[i, di] = be.max(time_series[min_idx:max_idx, di])

    return envelope_down, envelope_up


def lb_envelope(ts, radius=1, be=None):
    r"""Compute time series envelope as required by LB_Keogh.

    LB_Keogh was originally presented in [1]_.

    Parameters
    ----------
    ts : array-like, shape=(sz, d) or (sz,)
        Time series for which the envelope should be computed.
        If shape is (sz,), the time series is assumed to be univariate.
    radius : int (default: 1)
        Radius to be used for the envelope generation (the envelope at time
        index i will be generated based on all observations from the time series
        at indices comprised between i-radius and i+radius).
    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    Returns
    -------
    envelope_down : array-like, shape=(sz, d)
        Lower-side of the envelope.
    envelope_up : array-like, shape=(sz, d)
        Upper-side of the envelope.

    Examples
    --------
    >>> ts1 = [1, 2, 3, 2, 1]
    >>> env_low, env_up = lb_envelope(ts1, radius=1)
    >>> env_low
    array([[1.],
           [1.],
           [2.],
           [1.],
           [1.]])
    >>> env_up
    array([[2.],
           [3.],
           [3.],
           [3.],
           [2.]])

    See also
    --------
    lb_keogh : Compute LB_Keogh similarity

    References
    ----------
    .. [1] Keogh, E. Exact indexing of dynamic time warping. In International
       Conference on Very Large Data Bases, 2002. pp 406-417.
    """
    be = instantiate_backend(be, ts)
    ts = to_time_series(ts, be=be)
    if be.is_numpy:
        return _njit_lb_envelope(ts, radius=radius)
    return _lb_envelope(ts, radius=radius, be=be)
