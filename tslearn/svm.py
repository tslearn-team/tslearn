"""
The :mod:`tslearn.svm` module contains Support Vector Classifier (SVC) and Support Vector Regressor (SVR) models
for time series.
"""

from sklearn.svm import SVC as BaseSVC, SVR as BaseSVR
import numpy

from tslearn.metrics import cdist_gak, gamma_soft_dtw
from tslearn.utils import to_time_series_dataset


__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def _prepare_ts_datasets_sklearn(X):
    """Prepare time series datasets for sklearn.

    Examples
    --------
    >>> X = to_time_series_dataset([[1, 2, 3], [2, 2, 3]])
    >>> _prepare_ts_datasets_sklearn(X).shape
    (2, 3)
    """
    sklearn_X = to_time_series_dataset(X)
    n_ts, sz, d = sklearn_X.shape
    return sklearn_X.reshape((n_ts, -1))


def _kernel_func_gak(sz, d, gamma):
    if gamma == "auto":
        gamma = 1.
    return lambda x, y: cdist_gak(x.reshape((-1, sz, d)), y.reshape((-1, sz, d)), sigma=numpy.sqrt(gamma / 2.))


class TimeSeriesSVC(BaseSVC):
    """Time-series specific Support Vector Classifier.

    Parameters
    ----------
    sz : int
        Time series length
    d : int
        Time series dimensionality
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    kernel : string, optional (default='gak')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'gak' or a kernel accepted by ``sklearn.svm.SVC``.
         If none is given, 'gak' will be used. If a callable is given it is
         used to pre-compute the kernel matrix from data matrices; that matrix
         should be an array of shape ``(n_samples, n_samples)``.
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : float, optional (default='auto')
        Kernel coefficient for 'gak', 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then:
        - for 'gak' kernel, it is computed based on a sampling of the training set (cf `tslearn.metrics.gamma_soft_dtw`)
        - for other kernels (eg. 'rbf'), 1/n_features will be used.
    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    probability : boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.
    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.
    cache_size : float, optional
        Specify the size of the kernel cache (in MB).
    class_weight : {dict, 'balanced'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.
    decision_function_shape : 'ovo', 'ovr', default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2).
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
    n_support_ : array-like, dtype=int32, shape = [n_class]
        Number of support vectors for each class.
    dual_coef_ : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the section about multi-class classification in the
        SVM section of the User Guide of ``sklearn`` for details.
    coef_ : array, shape = [n_class-1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.
    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=64, d=2, n_blobs=2)
    >>> clf = TimeSeriesSVC(sz=64, d=2, kernel="gak", gamma="auto", probability=True)
    >>> clf.fit(X, y).predict(X).shape
    (20,)
    >>> sv = clf.support_vectors_time_series_(X)
    >>> len(sv)  # should be equal to the number of classes in the classification problem
    2
    >>> sv[0].shape  # doctest: +ELLIPSIS
    (..., 64, 2)
    >>> sum([sv_i.shape[0] for sv_i in sv]) == clf.n_support_.sum()
    True
    >>> clf.decision_function(X).shape
    (20,)
    >>> clf.predict_log_proba(X).shape
    (20, 2)
    >>> clf.predict_proba(X).shape
    (20, 2)

    References
    ----------
    Fast Global Alignment Kernels.
    Marco Cuturi.
    ICML 2011.
    """
    def __init__(self, sz, d, C=1.0, kernel="gak", degree=3, gamma="auto", coef0=0.0, shrinking=True,
                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape="ovr", random_state=None):
        self.sz = sz
        self.d = d
        if kernel == "gak":
            kernel = _kernel_func_gak(sz=sz, d=d, gamma=gamma)
        super(TimeSeriesSVC, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                                            shrinking=shrinking, probability=probability, tol=tol,
                                            cache_size=cache_size, class_weight=class_weight, verbose=verbose,
                                            max_iter=max_iter, decision_function_shape=decision_function_shape,
                                            random_state=random_state)

    def support_vectors_time_series_(self, X):
        X_ = to_time_series_dataset(X)
        sv = []
        idx_start = 0
        for cl in range(len(self.n_support_)):
            indices = self.support_[idx_start:idx_start + self.n_support_[cl]]
            sv.append(X_[indices])
            idx_start += self.n_support_[cl]
        return sv

    def fit(self, X, y, sample_weight=None):
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        if self.kernel == "gak" and self.gamma == "auto":
            self.gamma = gamma_soft_dtw(to_time_series_dataset(X))
            self.kernel = _kernel_func_gak(sz=self.sz, d=self.d, gamma=self.gamma)
        return super(TimeSeriesSVC, self).fit(sklearn_X, y, sample_weight=sample_weight)

    def predict(self, X):
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return super(TimeSeriesSVC, self).predict(sklearn_X)

    def decision_function(self, X):
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return super(TimeSeriesSVC, self).decision_function(sklearn_X)

    def predict_log_proba(self, X):
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return super(TimeSeriesSVC, self).predict_log_proba(sklearn_X)

    def predict_proba(self, X):
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return super(TimeSeriesSVC, self).predict_proba(sklearn_X)

    def score(self, X, y, sample_weight=None):
        return super(TimeSeriesSVC, self).score(X, y, sample_weight=sample_weight)


class TimeSeriesSVR(BaseSVR):
    """Time-series specific Support Vector Regressor.

    Parameters
    ----------
    sz : int
        Time series length
    d : int
        Time series dimensionality
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.
    epsilon : float, optional (default=0.1)
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.
    kernel : string, optional (default='gak')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'gak' or a kernel accepted by ``sklearn.svm.SVC``.
         If none is given, 'gak' will be used. If a callable is given it is
         used to pre-compute the kernel matrix from data matrices; that matrix
         should be an array of shape ``(n_samples, n_samples)``.
    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : float, optional (default='auto')
        Kernel coefficient for 'gak', 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then:
        - for 'gak' kernel, it is computed based on a sampling of the training set (cf `tslearn.metrics.gamma_soft_dtw`)
        - for other kernels (eg. 'rbf'), 1/n_features will be used.
    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.
    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.
    cache_size : float, optional
        Specify the size of the kernel cache (in MB).
    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
    dual_coef_ : array, shape = [1, n_SV]
        Coefficients of the support vector in the decision function.
    coef_ : array, shape = [1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is readonly property derived from `dual_coef_` and
        `support_vectors_`.
    intercept_ : array, shape = [1]
        Constants in decision function.
    sample_weight : array-like, shape = [n_samples]
        Individual weights for each sample

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=64, d=2, n_blobs=2)
    >>> import numpy
    >>> y = y.astype(numpy.float) + numpy.random.randn(20) * .1
    >>> reg = TimeSeriesSVR(sz=64, d=2, kernel="gak", gamma="auto")
    >>> reg.fit(X, y).predict(X).shape
    (20,)
    >>> sv = reg.support_vectors_time_series_(X)
    >>> sv.shape  # doctest: +ELLIPSIS
    (..., 64, 2)
    >>> sv.shape[0] <= 20
    True


    References
    ----------
    Fast Global Alignment Kernels.
    Marco Cuturi.
    ICML 2011.
    """
    def __init__(self, sz, d, kernel="gak", degree=3, gamma="auto", coef0=0.0, tol=0.001, C=1.0, epsilon=0.1,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        self.sz = sz
        self.d = d
        if kernel == "gak":
            kernel = _kernel_func_gak(sz=sz, d=d, gamma=gamma)
        super(TimeSeriesSVR, self).__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
                                            shrinking=shrinking, tol=tol, cache_size=cache_size, epsilon=epsilon,
                                            verbose=verbose, max_iter=max_iter)

    def support_vectors_time_series_(self, X):
        X_ = to_time_series_dataset(X)
        return X_[self.support_]

    def fit(self, X, y, sample_weight=None):
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        if self.kernel == "gak" and self.gamma == "auto":
            self.gamma = gamma_soft_dtw(to_time_series_dataset(X))
            self.kernel = _kernel_func_gak(sz=self.sz, d=self.d, gamma=self.gamma)
        return super(TimeSeriesSVR, self).fit(sklearn_X, y, sample_weight=sample_weight)

    def predict(self, X):
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return super(TimeSeriesSVR, self).predict(sklearn_X)

    def score(self, X, y, sample_weight=None):
        return super(TimeSeriesSVR, self).score(X, y, sample_weight=sample_weight)
