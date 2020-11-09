from sklearn.svm import SVC, SVR
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import deprecated
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
import numpy

from ..metrics import cdist_gak, gamma_soft_dtw, VARIABLE_LENGTH_METRICS
from ..utils import to_time_series_dataset, check_dims, to_sklearn_dataset
from ..bases import TimeSeriesBaseEstimator

import warnings

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class TimeSeriesSVMMixin:
    def _preprocess_sklearn(self, X, y=None, fit_time=False):
        force_all_finite = self.kernel not in VARIABLE_LENGTH_METRICS
        if y is None:
            X = check_array(X, allow_nd=True,
                            force_all_finite=force_all_finite)
        else:
            X, y = check_X_y(X, y, allow_nd=True,
                             force_all_finite=force_all_finite)
        X = to_time_series_dataset(X)

        if fit_time:
            self._X_fit = X
            if self.gamma == "auto":
                self.gamma_ = gamma_soft_dtw(X)
            else:
                self.gamma_ = self.gamma
            self.classes_ = numpy.unique(y)
        else:
            check_is_fitted(self, ['svm_estimator_', '_X_fit'])
            X = check_dims(
                X,
                X_fit_dims=self._X_fit.shape,
                extend=True,
                check_n_features_only=(self.kernel in VARIABLE_LENGTH_METRICS)
            )

        if self.kernel in VARIABLE_LENGTH_METRICS:
            assert self.kernel == "gak"
            self.estimator_kernel_ = "precomputed"
            if fit_time:
                sklearn_X = cdist_gak(X,
                                      sigma=numpy.sqrt(self.gamma_ / 2.),
                                      n_jobs=self.n_jobs, 
                                      verbose=self.verbose)
            else:
                sklearn_X = cdist_gak(X,
                                      self._X_fit,
                                      sigma=numpy.sqrt(self.gamma_ / 2.),
                                      n_jobs=self.n_jobs,
                                      verbose=self.verbose)
        else:
            self.estimator_kernel_ = self.kernel
            sklearn_X = to_sklearn_dataset(X)

        if y is None:
            return sklearn_X
        else:
            return sklearn_X, y


class TimeSeriesSVC(TimeSeriesSVMMixin, ClassifierMixin,
                    TimeSeriesBaseEstimator):
    """Time-series specific Support Vector Classifier.

    Parameters
    ----------
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

        - for 'gak' kernel, it is computed based on a sampling of the training
          set (cf :ref:`tslearn.metrics.gamma_soft_dtw <fun-tslearn.metrics.gamma_soft_dtw>`)
        - for other kernels (eg. 'rbf'), 1/n_features will be used.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    probability : boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.
        Also, probability estimates are not guaranteed to match predict output.
        See our :ref:`dedicated user guide section <kernels-ml>`
        for more details.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    cache_size : float, optional (default=200.0)
        Specify the size of the kernel cache (in MB).

    class_weight : {dict, 'balanced'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for GAK cross-similarity matrix
        computations.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    verbose : int, default: 0
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
        
    support_vectors_ : list of arrays of shape [n_SV, sz, d]
        List of support vectors in tslearn dataset format, one array per class

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

    svm_estimator_ : sklearn.svm.SVC
        The underlying sklearn estimator

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=64, d=2, n_blobs=2)
    >>> clf = TimeSeriesSVC(kernel="gak", gamma="auto", probability=True)
    >>> clf.fit(X, y).predict(X).shape
    (20,)
    >>> sv = clf.support_vectors_
    >>> len(sv)  # should be equal to the nr of classes in the clf problem
    2
    >>> sv[0].shape  # doctest: +ELLIPSIS
    (..., 64, 2)
    >>> sv_sum = sum([sv_i.shape[0] for sv_i in sv])
    >>> sv_sum == clf.svm_estimator_.n_support_.sum()
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
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, n_jobs=None, verbose=0, max_iter=-1,
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
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

    @property
    def n_iter_(self):
        warnings.warn('n_iter_ is always set to 1 for TimeSeriesSVC, since '
                      'it is non-trivial to access the underlying libsvm')
        return 1

    @deprecated('The use of '
                '`support_vectors_time_series_` is deprecated in '
                'tslearn v0.4 and will be removed in v0.6. Use '
                '`support_vectors_` property instead.')
    def support_vectors_time_series_(self, X=None):
        warnings.warn('The use of '
                      '`support_vectors_time_series_` is deprecated in '
                      'tslearn v0.4 and will be removed in v0.6. Use '
                      '`support_vectors_` property instead.')
        check_is_fitted(self, '_X_fit')
        return self._X_fit[self.svm_estimator_.support_]

    @property
    def support_vectors_(self):
        check_is_fitted(self, '_X_fit')
        sv = []
        idx_start = 0
        for cl in range(len(self.svm_estimator_.n_support_)):
            idx_end = idx_start + self.svm_estimator_.n_support_[cl]
            indices = self.svm_estimator_.support_[idx_start:idx_end]
            sv.append(self._X_fit[indices])
            idx_start += self.svm_estimator_.n_support_[cl]
        return sv

    def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
            
        y : array-like of shape=(n_ts, )
            Time series labels.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights force the 
            classifier to put more emphasis on these points.
        """
        sklearn_X, y = self._preprocess_sklearn(X, y, fit_time=True)

        self.svm_estimator_ = SVC(
            C=self.C, kernel=self.estimator_kernel_, degree=self.degree,
            gamma=self.gamma_, coef0=self.coef0, shrinking=self.shrinking,
            probability=self.probability, tol=self.tol,
            cache_size=self.cache_size, class_weight=self.class_weight,
            verbose=self.verbose, max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            random_state=self.random_state
        )
        self.svm_estimator_.fit(sklearn_X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):
        """Predict class for a given set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, ) or (n_ts, n_classes), depending on the shape
        of the label vector provided at training time.
            Index of the cluster each sample belongs to or class probability
            matrix, depending on what was provided at training time.
        """
        sklearn_X = self._preprocess_sklearn(X, fit_time=False)
        return self.svm_estimator_.predict(sklearn_X)

    def decision_function(self, X):
        """Evaluates the decision function for the samples in X.
        
        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
        
        Returns
        -------
        ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
            If decision_function_shape='ovr', the shape is (n_samples,
            n_classes)."""
        sklearn_X = self._preprocess_sklearn(X, fit_time=False)
        return self.svm_estimator_.decision_function(sklearn_X)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for a given set of time series.
        
        Note that probability estimates are not guaranteed to match predict 
        output.
        See our :ref:`dedicated user guide section <kernels-ml>`
        for more details.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_classes),
            Class probability matrix.
        """
        sklearn_X = self._preprocess_sklearn(X, fit_time=False)
        return self.svm_estimator_.predict_log_proba(sklearn_X)

    def predict_proba(self, X):
        """Predict class probability for a given set of time series.
        
        Note that probability estimates are not guaranteed to match predict 
        output.
        See our :ref:`dedicated user guide section <kernels-ml>`
        for more details.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, n_classes),
            Class probability matrix.
        """
        sklearn_X = self._preprocess_sklearn(X, fit_time=False)
        return self.svm_estimator_.predict_proba(sklearn_X)

    def _more_tags(self):
        return {'non_deterministic': True, 'allow_nan': True,
                'allow_variable_length': True}


class TimeSeriesSVR(TimeSeriesSVMMixin, RegressorMixin,
                    TimeSeriesBaseEstimator):
    """Time-series specific Support Vector Regressor.

    Parameters
    ----------
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

        - for 'gak' kernel, it is computed based on a sampling of the training
          set (cf :ref:`tslearn.metrics.gamma_soft_dtw <fun-tslearn.metrics.gamma_soft_dtw>`)
        - for other kernels (eg. 'rbf'), 1/n_features will be used.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    epsilon : float, optional (default=0.1)
         Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
         within which no penalty is associated in the training loss function
         with points predicted within a distance epsilon from the actual
         value.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    cache_size :  float, optional (default=200.0)
        Specify the size of the kernel cache (in MB).

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for GAK cross-similarity matrix
        computations.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.

    verbose : int, default: 0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.
        
    support_vectors_ : array of shape [n_SV, sz, d]
        Support vectors in tslearn dataset format

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

    svm_estimator_ : sklearn.svm.SVR
        The underlying sklearn estimator

    Examples
    --------
    >>> from tslearn.generators import random_walk_blobs
    >>> X, y = random_walk_blobs(n_ts_per_blob=10, sz=64, d=2, n_blobs=2)
    >>> import numpy
    >>> y = y.astype(numpy.float) + numpy.random.randn(20) * .1
    >>> reg = TimeSeriesSVR(kernel="gak", gamma="auto")
    >>> reg.fit(X, y).predict(X).shape
    (20,)
    >>> sv = reg.support_vectors_
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
                 coef0=0.0, tol=0.001, epsilon=0.1, shrinking=True,
                 cache_size=200, n_jobs=None, verbose=0, max_iter=-1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter

    @property
    def n_iter_(self):
        warnings.warn('n_iter_ is always set to 1 for TimeSeriesSVR, since '
                      'it is non-trivial to access the underlying libsvm')
        return 1

    @deprecated('The use of '
                '`support_vectors_time_series_` is deprecated in '
                'tslearn v0.4 and will be removed in v0.6. Use '
                '`support_vectors_` property instead.')
    def support_vectors_time_series_(self, X=None):
        warnings.warn('The use of '
                      '`support_vectors_time_series_` is deprecated in '
                      'tslearn v0.4 and will be removed in v0.6. Use '
                      '`support_vectors_` property instead.')
        check_is_fitted(self, '_X_fit')
        return self._X_fit[self.svm_estimator_.support_]

    @property
    def support_vectors_(self):
        check_is_fitted(self, '_X_fit')
        return self._X_fit[self.svm_estimator_.support_]

    def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.
            
        y : array-like of shape=(n_ts, )
            Time series labels.
            
        sample_weight : array-like of shape (n_samples,), default=None
            Per-sample weights. Rescale C per sample. Higher weights force the 
            classifier to put more emphasis on these points.
        """
        sklearn_X, y = self._preprocess_sklearn(X, y, fit_time=True)

        self.svm_estimator_ = SVR(
            C=self.C, kernel=self.estimator_kernel_, degree=self.degree,
            gamma=self.gamma_, coef0=self.coef0, shrinking=self.shrinking,
            tol=self.tol, cache_size=self.cache_size,
            verbose=self.verbose, max_iter=self.max_iter
        )
        self.svm_estimator_.fit(sklearn_X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        """Predict class for a given set of time series.

        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        Returns
        -------
        array of shape=(n_ts, ) or (n_ts, dim_output), depending on the shape
        of the target vector provided at training time.
            Predicted targets
        """
        sklearn_X = self._preprocess_sklearn(X, fit_time=False)
        return self.svm_estimator_.predict(sklearn_X)

    def _more_tags(self):
        return {'non_deterministic': True, 'allow_nan': True,
                'allow_variable_length': True}
