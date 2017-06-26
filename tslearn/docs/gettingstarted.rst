Getting started
===============

This tutorial will guide you to format your first time series data, import standard datasets, and manipulate them
using dedicated machine learning algorithms.

Time series format
------------------

First, let us have a look at what `tslearn` time series format is. To do so, we will use the ``to_time_series`` utility
from ``tslearn.utils`` module:

.. code-block:: python
  
    >>> from tslearn.utils import to_time_series
    >>> my_first_time_series = [1, 3, 4, 2]
    >>> formatted_time_series = to_time_series(my_first_time_series)
    >>> print(formatted_time_series.shape)
    (4, 1)

In `tslearn`, a time series is nothing more than a two-dimensional `numpy` array with its first dimension corresponding
to the time axis and the second one being the feature dimensionality (1 by default).

Then, if we want to manipulate sets of time series, we can cast them to three-dimensional arrays (if all time series
are the same length) or a list of time series, using ``to_time_series_dataset``.

.. code-block:: python

    >>> from tslearn.utils import to_time_series_dataset
    >>> my_first_time_series = [1, 3, 4, 2]
    >>> my_second_time_series = [1, 2, 4, 2]
    >>> formatted_dataset = to_time_series_dataset([my_first_time_series, my_second_time_series])
    >>> print(formatted_dataset.shape)
    (2, 4, 1)
    >>> my_third_time_series = [1, 2, 4, 2, 2]
    >>> formatted_dataset = to_time_series_dataset([my_first_time_series,
                                                    my_second_time_series,
                                                    my_third_time_series], equal_size=False)
    >>> print(type(formatted_dataset))
    list
    >>> print(len(formatted_dataset))
    3
    >>> print(formatted_dataset[0].shape)
    (4, 1)
    >>> print(formatted_dataset[1].shape)
    (4, 1)
    >>> print(formatted_dataset[2].shape)
    (5, 1)


Importing standard time series datasets
---------------------------------------

If you aim at experimenting with standard time series datasets, you should have a look at the
:ref:`tslearn.datasets <mod-datasets>` module.

.. code-block:: python

    >>> from tslearn.datasets import UCR_UEA_datasets
    >>> X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("TwoPatterns")
    >>> print(X_train.shape)
    (1000, 128, 1)
    >>> print(y_train.shape)
    (1000,)

When working with time series datasets, it is can be useful to rescale time series using tools from the
:ref:`tslearn.preprocessing <mod-preprocessing>` module.

Playing with your data
----------------------

Once your data is loaded and formatted according to `tslearn` standards, the next step is to feed machine learning
models with it. Most `tslearn` models inherit from `scikit-learn` base classes, hence interacting with them is very
similar to interacting with a `scikit-learn` model, except that datasets are not two-dimensional arrays, but rather
`tslearn` time series datasets (`i.e.` three-dimensional arrays or lists of two-dimensional arrays).

.. code-block:: python

    >>> from tslearn.clustering import TimeSeriesKMeans
    >>> km = TimeSeriesKMeans(n_clusters=3, metric="dtw")
    >>> km.fit(X_train)

As seen above, one key parameter when applying machine learning methods to time series datasets is the metric to be
used. You can learn more about it in the :ref:`dedicated section <mod-metrics>` of this documentation.
