import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.distance import pdist, squareform
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from tslearn.bases import BaseModelPackage, TimeSeriesBaseEstimator
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import check_dims

stumpy_msg = ('stumpy is not installed, stumpy features will not be'
              'supported.\n Install stumpy to use stumpy features:'
              'https://stumpy.readthedocs.io/en/latest/')

try:
    import stumpy
except ImportError:
    STUMPY_INSTALLED = False
else:
    STUMPY_INSTALLED = True

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _series_to_segments(time_series, segment_size):
    """Transforms a time series (or a set of time series) into an array of its
    segments of length segment_size.

    Examples
    --------
    >>> from tslearn.utils import to_time_series
    >>> time_series = to_time_series([1, 2, 3, 4, 5])
    >>> segments = _series_to_segments(time_series, segment_size=2)
    >>> segments.shape
    (4, 2, 1)
    >>> segments[:, :, 0]
    array([[1., 2.],
           [2., 3.],
           [3., 4.],
           [4., 5.]])
    >>> from tslearn.utils import to_time_series_dataset
    >>> dataset = to_time_series_dataset([time_series])
    >>> dataset.shape
    (1, 5, 1)
    >>> segments = _series_to_segments(dataset, segment_size=2)
    >>> segments.shape
    (4, 2, 1)
    >>> segments[:, :, 0]
    array([[1., 2.],
           [2., 3.],
           [3., 4.],
           [4., 5.]])
    """
    if time_series.ndim == 3:
        l_segments = [_series_to_segments(ts, segment_size)
                      for ts in time_series]
        return np.vstack(l_segments)
    elem_size = time_series.strides[0]
    segments = as_strided(
        time_series,
        strides=(elem_size, elem_size, time_series.strides[1]),
        shape=(time_series.shape[0] - segment_size + 1,
               segment_size, time_series.shape[1]),
        writeable=False
    )
    return segments


class MatrixProfile(TransformerMixin,
                    BaseModelPackage,
                    TimeSeriesBaseEstimator):
    """Matrix Profile transformation.

    Matrix Profile was originally presented in [1]_.

    Parameters
    ----------
    subsequence_length : int (default: 1)
        Length of the subseries (also called window size) to be used for
        subseries distance computations.

    implementation : str (default: "numpy")
        Matrix profile implementation to use.
        Defaults to "numpy" to use the pure numpy version.
        All the available implementations are ["numpy", "stump", "gpu_stump"].

        "stump" and "gpu_stump" are both implementations from the stumpy
        python library, the latter requiring a GPU.
        Stumpy is a library for efficiently computing the matrix profile which
        is optimized for speed, performance and memory.
        See [2]_ for the documentation.
        "numpy" is the default pure numpy implementation and does not require
        stumpy to be installed.

    scale: bool (default: True)
         Whether input data should be scaled for each feature of each time
         series to have zero mean and unit variance.
         Default for this parameter is set to `True` to match the standard
         matrix profile setup.

    Examples
    --------
    >>> time_series = [0., 1., 3., 2., 9., 1., 14., 15., 1., 2., 2., 10., 7.]
    >>> ds = [time_series]
    >>> mp = MatrixProfile(subsequence_length=4, scale=False)
    >>> mp.fit_transform(ds)[0, :, 0]  # doctest: +ELLIPSIS
    array([ 6.85...,  1.41...,  6.16...,  7.93..., 11.40...,
           13.56..., 18.  ..., 13.96...,  1.41...,  6.16...])

    References
    ----------
    .. [1] C. M. Yeh, Y. Zhu, L. Ulanova, N.Begum et al.
       Matrix Profile I: All Pairs Similarity Joins for Time Series: A
       Unifying View that Includes Motifs, Discords and Shapelets.
       ICDM 2016.

    .. [2] STUMPY documentation https://stumpy.readthedocs.io/en/latest/

    """

    def __init__(
        self, subsequence_length=1, implementation="numpy", scale=True
    ):
        self.subsequence_length = subsequence_length
        self.scale = scale
        self.implementation = implementation

    def _is_fitted(self):
        check_is_fitted(self, '_X_fit_dims')
        return True

    def _fit(self, X, y=None):
        self._X_fit_dims = X.shape
        return self

    def fit(self, X, y=None):
        """Fit a Matrix Profile representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        MatrixProfile
            self
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)

    def _transform(self, X, y=None):
        n_ts, sz, d = X.shape

        if d > 1:
            raise NotImplementedError("We currently don't support using "
                                      "multi-dimensional matrix profiles "
                                      "from the stumpy library.")

        output_size = sz - self.subsequence_length + 1
        X_transformed = np.empty((n_ts, output_size, 1))

        if self.implementation == "stump":
            if not STUMPY_INSTALLED:
                raise ImportError(stumpy_msg)

            for i_ts in range(n_ts):
                result = stumpy.stump(
                    T_A=X[i_ts, :, 0].ravel(),
                    m=self.subsequence_length)
                X_transformed[i_ts, :, 0] = result[:, 0].astype(np.float)

        elif self.implementation == "gpu_stump":
            if not STUMPY_INSTALLED:
                raise ImportError(stumpy_msg)

            for i_ts in range(n_ts):
                result = stumpy.gpu_stump(
                    T_A=X[i_ts, :, 0].ravel(),
                    m=self.subsequence_length)
                X_transformed[i_ts, :, 0] = result[:, 0].astype(np.float)

        elif self.implementation == "numpy":
            scaler = TimeSeriesScalerMeanVariance()
            band_width = int(np.ceil(self.subsequence_length / 4))
            for i_ts in range(n_ts):
                segments = _series_to_segments(X[i_ts], self.subsequence_length)
                if self.scale:
                    segments = scaler.fit_transform(segments)
                n_segments = segments.shape[0]
                segments_2d = segments.reshape((-1, self.subsequence_length * d))
                dists = squareform(pdist(segments_2d, "euclidean"))
                band = (np.tri(n_segments, n_segments,
                            band_width, dtype=np.bool) &
                        ~np.tri(n_segments, n_segments,
                                -(band_width + 1), dtype=np.bool))
                dists[band] = np.inf
                X_transformed[i_ts] = dists.min(axis=1, keepdims=True)

        else:
            available_implementations = ["numpy", "stump", "gpu_stump"]
            raise ValueError(
                'This "{}" matrix profile implementation is not'
                ' recognized. Available implementations are {}.'
                .format(self.implementation,
                        available_implementations)
            )

        return X_transformed

    def transform(self, X, y=None):
        """Transform a dataset of time series into its Matrix Profile
         representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, output_size, 1)
            Matrix-Profile-Transformed dataset. `ouput_size` is equal to
            `sz - subsequence_length + 1`
        """
        self._is_fitted()
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X, X_fit_dims=self._X_fit_dims,
                       check_n_features_only=True)
        return self._transform(X, y)

    def fit_transform(self, X, y=None, **fit_params):
        """Transform a dataset of time series into its Matrix Profile
         representation.

        Parameters
        ----------
        X : array-like of shape (n_ts, sz, d)
            Time series dataset

        Returns
        -------
        numpy.ndarray of shape (n_ts, output_size, 1)
            Matrix-Profile-Transformed dataset. `ouput_size` is equal to
            `sz - subsequence_length + 1`
        """
        X = check_array(X, allow_nd=True, force_all_finite=False)
        X = check_dims(X)
        return self._fit(X)._transform(X)

    def _more_tags(self):
        return {'allow_nan': True, 'allow_variable_length': True}
