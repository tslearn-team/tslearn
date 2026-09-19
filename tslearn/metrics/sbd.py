import numpy as np

from numpy.linalg import norm
from numpy.fft import fft, ifft


def sbd(s1, s2):
    """Compute the Shape-based distance (SBD) measure between two time series and return the distance between the given
    time series and the shift of s2 that maximizes

    The computation of SBD is based on cross-correlation. Cross-correlation is a statistical
    measure to determine the similarity of two sequences ~s1 and ~s2, even if they are not properly aligned.
    To achieve shift-invariance, cross-correlation keeps ~s1 static and slides ~s2 over ~s1 to compute
    their inner product for each shift s of ~s2. SBD is then calculated as 1 - the coefficient normalized
    cross-correlation of the optimal shift of s2, where the optimal shift of s2 is the one that maximizes the
    coefficient normalized cross-correlation.

    SBD takes values between 0 to 2, with 0 indicating perfect similarity for time-series sequences.
    SBD was originally presented in [1]_.
    Parameters
    ----------
    s1
        A time series.
    s2
        Another time series.

    Returns
    -------
    float
        The Shape-based distance (SBD) measure between s1 and s2
    numpy.ndarray
        The shift of s2

    Examples
    --------
    >>> sbd_dist, s2_shift = sbd([1, 2, 3], [1, 2, 3])
    >>> sbd_dist
    0.0
    >>> type(s2_shift)
    <class 'numpy.ndarray'>
    >>> s2_shift
    array([1, 2, 3])

    References
    ----------
    .. [1] John Paparrizos and Luis Gravano. 2015. K-Shape: Efficient and Accurate Clustering of Time Series. In
    Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data (SIGMOD '15).
    """
    # Get all possible shifts and their coefficient normalized cross-correlation
    ncc = _ncc_c(s1, s2)
    # Choose the shift with the highest coefficient normalized cross-correlation
    idx = ncc.argmax()
    # Calculate SBD according to Equation 9
    dist = 1 - ncc[idx]
    # Pad the chosen s2_shift with zeros
    s2_shift = _roll_zeropad(s2, (idx + 1) - max(len(s1), len(s2)))

    return dist, s2_shift


def _ncc_c(s1, s2):
    den = np.array(norm(s1) * norm(s2))
    den[den == 0] = np.Inf

    s1_len = len(s1)
    # Performance optimization: Calculate the next power-of-two, fft will then pad with zeros to reach fft_size
    fft_size = 1 << (2 * s1_len - 1).bit_length()
    # Equation 12
    cc = ifft(fft(s1, fft_size) * np.conj(fft(s2, fft_size)))
    cc = np.concatenate((cc[-(s1_len - 1):], cc[:s1_len]))
    return np.real(cc) / den


def _roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n - shift), axis))
        res = np.concatenate((a.take(np.arange(n - shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n - shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n - shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res
