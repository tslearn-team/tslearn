from joblib import Parallel, delayed

from tslearn.backend import instantiate_backend
from tslearn.utils import to_time_series_dataset

__author__ = "Romain Tavenard romain.tavenard[at]univ-rennes2.fr"


def _cdist_generic(
    dist_fun,
    dataset1,
    dataset2,
    n_jobs,
    verbose,
    compute_diagonal=True,
    dtype=float,
    be=None,
    *args,
    **kwargs
):
    """Compute cross-similarity matrix with joblib parallelization for a given
    similarity function.

    Parameters
    ----------
    dist_fun : function
        Similarity function to be used.

    dataset1 : array-like, shape=(n_ts1, sz1, d) or (n_ts1, sz1) or (sz1,)
        A dataset of time series.
        If shape is (n_ts1, sz1), the dataset is composed of univariate time series.
        If shape is (sz1,), the dataset is composed of a unique univariate time series.

    dataset2 : None or array-like, shape=(n_ts2, sz2, d) or (n_ts2, sz2) or (sz2,) (default: None)
        Another dataset of time series. 
        If `None`, self-similarity of `dataset1` is returned.
        If shape is (n_ts2, sz2), the dataset is composed of univariate time series.
        If shape is (sz2,), the dataset is composed of a unique univariate time series.

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

    be : Backend object or string or None
        Backend. If `be` is an instance of the class `NumPyBackend` or the string `"numpy"`,
        the NumPy backend is used.
        If `be` is an instance of the class `PyTorchBackend` or the string `"pytorch"`,
        the PyTorch backend is used.
        If `be` is `None`, the backend is determined by the input arrays.
        See our :ref:`dedicated user-guide page <backend>` for more information.

    compute_diagonal : bool (default: True)
        Whether diagonal terms should be computed or assumed to be 0 in the
        self-similarity case. Used only if `dataset2` is `None`.

    *args and **kwargs :
        Optional additional parameters to be passed to the similarity function.


    Returns
    -------
    cdist : array-like, shape=(n_ts1, n_ts2)
        Cross-similarity matrix.
    """  # noqa: E501
    be = instantiate_backend(be, dataset1, dataset2)
    dataset1 = to_time_series_dataset(dataset1, dtype=dtype, be=be)

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = be.zeros((len(dataset1), len(dataset1)))
        indices = be.triu_indices(
            len(dataset1), k=0 if compute_diagonal else 1, m=len(dataset1)
        )

        matrix[indices] = be.array(
            Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
                delayed(dist_fun)(dataset1[i], dataset1[j], *args, **kwargs)
                for i in range(len(dataset1))
                for j in range(i if compute_diagonal else i + 1, len(dataset1))
            )
        )

        indices = be.tril_indices(len(dataset1), k=-1, m=len(dataset1))
        matrix[indices] = matrix.T[indices]

        return matrix
    else:
        dataset2 = to_time_series_dataset(dataset2, dtype=dtype, be=be)
        matrix = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
            delayed(dist_fun)(dataset1[i], dataset2[j], *args, **kwargs)
            for i in range(len(dataset1))
            for j in range(len(dataset2))
        )
        return be.reshape(be.array(matrix), (len(dataset1), -1))
