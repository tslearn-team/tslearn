"""
The :mod:`tslearn.svm` module contains Support Vector Classifier (SVC) and Support Vector Regressor (SVR) models
for time series.
"""

from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import numpy

from tslearn.metrics import cdist_gak, gamma_soft_dtw
from tslearn.utils import to_time_series_dataset, check_dims
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.validation import check_is_fitted

import warnings


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


class GAKKernel():
    def __init__(self, sz, d, gamma):
        self.sz = sz
        self.d = d
        if gamma == "auto":
            self.gamma = 1.
        else:
            self.gamma = gamma

    def __call__(self, x, y):
        return cdist_gak(
            x.reshape((-1, self.sz, self.d)),
            y.reshape((-1, self.sz, self.d)),
            sigma=numpy.sqrt(self.gamma / 2.)
        )


class TimeSeriesSVC(BaseEstimator, ClassifierMixin):
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
    >>> clf = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
    >>> clf.fit(X, y).predict(X).shape
    (20,)
    >>> sv = clf.support_vectors_time_series_(X)
    >>> len(sv)  # should be equal to the number of classes in the classification problem
    2
    >>> sv[0].shape  # doctest: +ELLIPSIS
    (..., 64, 2)
    >>> sum([sv_i.shape[0] for sv_i in sv]) == clf.svm_estimator_.n_support_.sum()
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
    def __init__(self, C=1.0, kernel="gak", degree=3, gamma="auto", coef0=0.0,
                 shrinking=True, probability=True, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape="ovr", random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

    @property
    def n_iter_(self):
        warnings.warn('n_iter_ is always 1 for TimeSeriesSVC, since '
                      'it is non-trivial to access from libsvm')
        return 1

    def _kernel_func_gak(self, x, y):
        return cdist_gak(x.reshape((-1, self.sz_, self.d_)),
                         y.reshape((-1, self.sz_, self.d_)),
                         sigma=numpy.sqrt(self.gamma_ / 2.))

    def support_vectors_time_series_(self, X):
        X_ = to_time_series_dataset(X)
        sv = []
        idx_start = 0
        for cl in range(len(self.svm_estimator_.n_support_)):
            indices = self.svm_estimator_.support_[idx_start:idx_start + self.svm_estimator_.n_support_[cl]]
            sv.append(X_[indices])
            idx_start += self.svm_estimator_.n_support_[cl]
        return sv

    def fit(self, X, y, sample_weight=None):
        X = check_array(X, allow_nd=True)
        y = column_or_1d(y, warn=True)
        X = check_dims(X, X_fit=None)

        self.X_fit_ = X
        self.classes_ = numpy.unique(y)

        _, sz, d = X.shape
        sklearn_X = _prepare_ts_datasets_sklearn(X)

        gamma = self.gamma
        kernel = self.kernel
        if gamma == "auto":
            gamma = gamma_soft_dtw(to_time_series_dataset(X))
        if kernel == "gak":
            kernel = GAKKernel(sz, d, gamma)

        self.svm_estimator_ = SVC(
            C=self.C, kernel=kernel, degree=self.degree,
            gamma=gamma, coef0=self.coef0, shrinking=self.shrinking,
            probability=self.probability, tol=self.tol,
            cache_size=self.cache_size, class_weight=self.class_weight,
            verbose=self.verbose, max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            random_state=self.random_state
        )
        self.svm_estimator_.fit(sklearn_X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        X = check_array(X, allow_nd=True)
        check_is_fitted(self, ['svm_estimator_', 'X_fit_'])
        X = check_dims(X, self.X_fit_)

        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return self.svm_estimator_.predict(sklearn_X)

    def decision_function(self, X):
        X = check_array(X, allow_nd=True)
        check_is_fitted(self, ['svm_estimator_', 'X_fit_'])
        X = check_dims(X, self.X_fit_)

        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return self.svm_estimator_.decision_function(sklearn_X)

    def predict_log_proba(self, X):
        X = check_array(X, allow_nd=True)
        check_is_fitted(self, ['svm_estimator_', 'X_fit_'])
        X = check_dims(X, self.X_fit_)

        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return self.svm_estimator_.predict_log_proba(sklearn_X)

    def predict_proba(self, X):
        X = check_array(X, allow_nd=True)
        check_is_fitted(self, ['svm_estimator_', 'X_fit_'])
        X = check_dims(X, self.X_fit_)

        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return self.svm_estimator_.predict_proba(sklearn_X)

    def score(self, X, y, sample_weight=None):
        X = check_array(X, allow_nd=True)
        y = column_or_1d(y, warn=True)
        check_is_fitted(self, ['svm_estimator_', 'X_fit_'])
        X = check_dims(X, X_fit=self.X_fit_)

        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return self.svm_estimator_.score(sklearn_X, y, 
                                         sample_weight=sample_weight)


class TimeSeriesSVR(BaseEstimator, RegressorMixin):
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
    >>> reg = TimeSeriesSVR(kernel="gak", gamma="auto")
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
    def __init__(self, C=1.0, kernel="gak", degree=3, gamma="auto", 
                 coef0=0.0, tol=0.001, epsilon=0.1,
                 shrinking=True, cache_size=200, verbose=False, max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter

    @property
    def n_iter_(self):
        warnings.warn('n_iter_ is always 1 for TimeSeriesSVR, since '
                      'it is non-trivial to access from libsvm')
        return 1

    def support_vectors_time_series_(self, X):
        X_ = to_time_series_dataset(X)
        return X_[self.svm_estimator_.support_]

    def fit(self, X, y, sample_weight=None):
        X = check_array(X, allow_nd=True)
        y = column_or_1d(y, warn=True)
        X = check_dims(X, X_fit=None)

        self.X_fit_ = X
        self.classes_ = numpy.unique(y)

        _, sz, d = X.shape
        sklearn_X = _prepare_ts_datasets_sklearn(X)

        gamma = self.gamma
        kernel = self.kernel
        if gamma == "auto":
            gamma = gamma_soft_dtw(to_time_series_dataset(X))
        if kernel == "gak":
            kernel = GAKKernel(sz, d, gamma)

        self.svm_estimator_ = SVR(
            C=self.C, kernel=kernel, degree=self.degree,
            gamma=gamma, coef0=self.coef0, shrinking=self.shrinking,
            tol=self.tol, cache_size=self.cache_size, 
            verbose=self.verbose, max_iter=self.max_iter
        )
        self.svm_estimator_.fit(sklearn_X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X = check_array(X, allow_nd=True)
        check_is_fitted(self, ['svm_estimator_', 'X_fit_'])
        X = check_dims(X, self.X_fit_)
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return  self.svm_estimator_.predict(sklearn_X)

    def score(self, X, y, sample_weight=None):
        X = check_array(X, allow_nd=True)
        check_is_fitted(self, ['svm_estimator_', 'X_fit_'])
        X = check_dims(X, self.X_fit_)
        sklearn_X = _prepare_ts_datasets_sklearn(X)
        return  self.svm_estimator_.score(sklearn_X, y, 
                                          sample_weight=sample_weight)

    def _more_tags(self):
        return {'non_deterministic': True}
