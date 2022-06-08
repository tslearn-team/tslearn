from .utils import _cdist_generic
from .cysax import cydist_sax

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def cdist_sax(dataset1, breakpoints_avg, size_fitted, dataset2=None,
              n_jobs=None, verbose=0):
    r"""Calculates a matrix of distances (MINDIST) on SAX-transformed data,
    as presented in [1]_. It is important to note that this function
    expects the timeseries in dataset1 and dataset2 to be normalized
    to each have zero mean and unit variance.

    Parameters
    ----------
    dataset1 : array-like
        A dataset of time series

    breakpoints_avg : array-like
        The breakpoints used to assign the alphabet symbols.

    size_fitted: int
        The original timesteps in the timeseries, before
        discretizing through SAX.

    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`__
        for more details.

    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.

    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix

    References
    ----------
    .. [1] Lin, Jessica, et al. "Experiencing SAX: a novel symbolic
           representation of time series." Data Mining and knowledge
           discovery 15.2 (2007): 107-144.

    """  # noqa: E501
    return _cdist_generic(cydist_sax, dataset1, dataset2, n_jobs, verbose,
                          False, int, breakpoints_avg, size_fitted)
