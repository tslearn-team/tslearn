Integration with other Python packages
--------------------------------------

``tslearn`` is a general-purpose Python machine learning library for time
series that offers tools for pre-processing and feature extraction as well as
dedicated models for clustering, classification and regression.
To ensure compatibility with more specific Python packages, we provide utilities
to convert data sets from and to other formats.

:func:`tslearn.utils.to_time_series_dataset` is a general function that
transforms an array-like object into a three-dimensional array of shape
``(n_ts, sz, d)`` with the following conventions:

- the fist axis is the sample axis, ``n_ts`` being the number of time series;
- the second axis is the time axis, ``sz`` being the maximum number of time points;
- the third axis is the dimension axis, ``d`` being the number of dimensions.

This is how a data set of time series is represented in ``tslearn``.

The following sections briefly explain how to transform a data set from
``tslearn`` to another supported Python package and vice versa.


scikit-learn
^^^^^^^^^^^^

`scikit-learn <https://scikit-learn.org>`_ is a popular Python package for
machine learning.
:func:`tslearn.utils.to_sklearn_dataset` converts a data set from ``tslearn``
format to ``scikit-learn`` format. To convert a data set from
``scikit-learn``, you can use :func:`tslearn.utils.to_time_series_dataset`.

.. code-block:: python

    >>> from tslearn.utils import to_sklearn_dataset
    >>> to_sklearn_dataset([[1, 2], [1, 4, 3]])
    array([[ 1.,  2., nan],
           [ 1.,  4.,  3.]])
    >>> to_time_series_dataset([[ 1.,  2., None], [ 1.,  4.,  3.]])
    array([[[ 1.],
        [ 2.],
        [nan]],

       [[ 1.],
        [ 4.],
        [ 3.]]])


pyts
^^^^

`pyts <https://pyts.readthedocs.io>`_ is a Python package dedicated to time
series classification.
:func:`tslearn.utils.to_pyts_dataset` and :func:`tslearn.utils.from_pyts_dataset`
allow users to convert a data set from ``tslearn`` format to ``pyts`` format
and vice versa.

.. code-block:: python

    >>> from tslearn.utils import from_pyts_dataset, to_pyts_dataset
    >>> from_pyts_dataset([[1, 2], [1, 4]])
    array([[[1],
            [2]],

           [[1],
            [4]]])

    >>> to_pyts_dataset([[[1], [2]], [[1], [4]]])
    array([[1., 2.],
           [1., 4.]])


seglearn
^^^^^^^^

`seglearn <https://dmbee.github.io/seglearn/>`_ is a python package for machine
learning time series or sequences.
:func:`tslearn.utils.to_seglearn_dataset` and :func:`tslearn.utils.from_seglearn_dataset`
allow users to convert a data set from ``tslearn`` format to ``seglearn`` format
and vice versa.

.. code-block:: python

    >>> from tslearn.utils import from_seglearn_dataset, to_seglearn_dataset
    >>> from_seglearn_dataset([[1, 2], [1, 4, 3]])
    array([[[ 1.],
            [ 2.],
            [nan]],

           [[ 1.],
            [ 4.],
            [ 3.]]])
    >>> to_seglearn_dataset([[[1], [2], [None]], [[1], [4], [3]]])
    array([array([[1.],
           [2.]]),
           array([[1.],
           [4.],
           [3.]])], dtype=object)


stumpy
^^^^^^

`stumpy <https://stumpy.readthedocs.io/>`_ is a powerful and scalable Python
library for computing a Matrix Profile, which can be used for a variety of time
series data mining tasks.
:func:`tslearn.utils.to_stumpy_dataset` and :func:`tslearn.utils.from_stumpy_dataset`
allow users to convert a data set from ``tslearn`` format to ``stumpy`` format
and vice versa.

.. code-block:: python

    >>> import numpy as np
    >>> from tslearn.utils import from_stumpy_dataset, to_stumpy_dataset
    >>> from_stumpy_dataset([np.array([1, 2]), np.array([1, 4, 3])])
    array([[[ 1.],
            [ 2.],
            [nan]],

           [[ 1.],
            [ 4.],
            [ 3.]]])
    >>> to_stumpy_dataset([[[1], [2], [None]], [[1], [4], [3]]])
    [array([1., 2.]), array([1., 4., 3.])]


sktime
^^^^^^

`sktime <https://alan-turing-institute.github.io/sktime/>`_ is a ``scikit-learn``
compatible Python toolbox for learning with time series.
:func:`tslearn.utils.to_sktime_dataset` and :func:`tslearn.utils.from_sktime_dataset`
allow users to convert a data set from ``tslearn`` format to ``sktime`` format
and vice versa.
``pandas`` is a required dependency to use these functions.

.. code-block:: python

    >>> import pandas as pd
    >>> from tslearn.utils import from_sktime_dataset, to_sktime_dataset
    >>> df = pd.DataFrame()
    >>> df["dim_0"] = [pd.Series([1, 2]), pd.Series([1, 4, 3])]
    >>> from_sktime_dataset(df)
    array([[[ 1.],
            [ 2.],
            [nan]],

           [[ 1.],
            [ 4.],
            [ 3.]]])
    >>> to_sktime_dataset([[[1], [2], [None]], [[1], [4], [3]]]).shape
    (2, 1)


pyflux
^^^^^^

`pyflux <https://pyflux.readthedocs.io>`_ is a library for time series analysis
and prediction.
:func:`tslearn.utils.to_pyflux_dataset` and :func:`tslearn.utils.from_pyflux_dataset`
allow users to convert a data set from ``tslearn`` format to ``pyflux`` format
and vice versa.
``pandas`` is a required dependency to use these functions.

.. code-block:: python

    >>> import pandas as pd
    >>> from tslearn.utils import from_pyflux_dataset, to_pyflux_dataset
    >>> df = pd.DataFrame([1, 2], columns=["dim_0"])
    >>> from_pyflux_dataset(df)
    array([[[1.],
            [2.]]])
    >>> to_pyflux_dataset([[[1], [2]]]).shape
    (2, 1)


tsfresh
^^^^^^^

`tsfresh <https://tsfresh.readthedocs.io>`_ iis a python package automatically
calculating a large number of time series characteristics.
:func:`tslearn.utils.to_tsfresh_dataset` and :func:`tslearn.utils.from_tsfresh_dataset`
allow users to convert a data set from ``tslearn`` format to ``tsfresh`` format
and vice versa.
``pandas`` is a required dependency to use these functions.

.. code-block:: python

    >>> import pandas as pd
    >>> from tslearn.utils import from_tsfresh_dataset, to_tsfresh_dataset
    >>> df = pd.DataFrame([[0, 0, 1.0],
    ...                    [0, 1, 2.0],
    ...                    [1, 0, 1.0],
    ...                    [1, 1, 4.0],
    ...                    [1, 2, 3.0]], columns=['id', 'time', 'dim_0'])
    >>> from_tsfresh_dataset(df)
    array([[[ 1.],
        [ 2.],
        [nan]],

       [[ 1.],
        [ 4.],
        [ 3.]]])
    >>> to_tsfresh_dataset([[[1], [2], [None]], [[1], [4], [3]]]).shape
    (5, 3)


cesium
^^^^^^

`cesium <http://cesium-ml.org>`_ is an open-source platform for time series inference.
:func:`tslearn.utils.to_cesium_dataset` and :func:`tslearn.utils.from_cesium_dataset`
allow users to convert a data set from ``tslearn`` format to ``cesium`` format
and vice versa.
``cesium`` is a required dependency to use these functions.

.. code-block:: python

    >>> from tslearn.utils import from_cesium_dataset, to_cesium_dataset
    >>> from cesium.data_management import TimeSeries
    >>> from_cesium_dataset([TimeSeries(m=[1, 2]), TimeSeries(m=[1, 4, 3])])
    array([[[ 1.],
            [ 2.],
            [nan]],

           [[ 1.],
            [ 4.],
            [ 3.]]])
    >>> len(to_cesium_dataset([[[1], [2], [None]], [[1], [4], [3]]]))
    2
