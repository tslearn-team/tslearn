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

Then, if we want to manipulate sets of time series, we can cast them to three-dimensional arrays, using
``to_time_series_dataset``. If time series from the set are not equal-sized, NaN values are appended to the shorter
ones and the shape of the resulting array is ``(n_ts, max_sz, d)`` where ``max_sz`` is the maximum of sizes for time
series in the set.

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
                                                    my_third_time_series])
    >>> print(formatted_dataset.shape)
    (3, 5, 1)


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

Note that when working with time series datasets, it can be useful to rescale time series using tools from the
:ref:`tslearn.preprocessing <mod-preprocessing>` module.

If you want to import other time series from text files, the expected format is:

* each line represents a single time series (and time series from a dataset are not forced to be the same length);
* in each line, modalities are separated by a `|` character (useless if you only have one modality in your data);
* in each modality, observations are sparated by a space character.

Here is an example of such a file storing two time series of dimension 2 (the first time series is of length 3 and
the second one is of length 2).

.. code-block:: csv

   1.0 0.0 2.5|3.0 2.0 1.0
   1.0 2.0|4.333 2.12

To read from / write to this format, have a look at the :ref:`tslearn.utils <mod-utils>` module:

.. code-block:: python

    >>> from tslearn.utils import save_timeseries_txt, load_timeseries_txt
    >>> time_series_dataset = load_timeseries_txt("path/to/your/file.txt")
    >>> save_timeseries_txt("path/to/another/file.txt", dataset_to_be_saved)

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
