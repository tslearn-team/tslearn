<!-- Our title -->
<div align="center">
    <span style="font-size: 50px;">tslearn </span>
</div>

<!-- Short description -->
<p align="center"> 
   "The machine learning toolkit for time series analysis in Python"
</p>

<!-- The badges -->
<p align="center">
    <a href="https://badge.fury.io/py/tslearn">
        <img alt="PyPI" src="https://badge.fury.io/py/tslearn.svg">
    </a>
    <a href="http://tslearn.readthedocs.io/en/latest/?badge=latest">
        <img alt="Documentation" src="https://readthedocs.org/projects/tslearn/badge/?version=latest">
    </a>
    <a href="https://travis-ci.org/rtavenar/tslearn">
        <img alt="Build" src="https://travis-ci.org/rtavenar/tslearn.svg?branch=master">
    </a>
    <a href="https://codecov.io/gh/rtavenar/tslearn">
        <img alt="Codecov" src="https://codecov.io/gh/rtavenar/tslearn/branch/master/graph/badge.svg">
    </a>
    <a href="https://pepy.tech/project/tslearn">
        <img alt="Downloads" src="https://pepy.tech/badge/tslearn">
    </a>
</p>

<!-- Draw horizontal rule -->
<hr>

<!-- Table of content -->

| Section | Description |
|-|-|
| [Installing](#installation) | Installing the dependencies and tslearn |
| [Getting started](#getting-started) | A quick introduction on how to use tslearn |
| [Documentation](#documentation) | A link to our API reference and a gallery of examples |
| [Contributing](#preparing-the-jsons) | A guide for heroes willing to contribute |
| [Citation](#referencing-tslearn) | A citation for tslearn for scholarly articles |

# Installation
There are different alternatives to install tslearn:
* PyPi: `python -m pip install tslearn`
* Conda: `conda install -c conda-forge tslearn`
* Git: `python -m pip install https://github.com/rtavenar/tslearn/archive/master.zip`

In order for the installation to be successful, the required dependencies must be installed. For a more detailed guide on how to install tslearn, please see the [Documentation](https://tslearn.readthedocs.io/en/latest/?badge=latest#installation)

# Getting started

## 1. Getting the data in the right format
tslearn expects the data to be 3D. The three dimensions correspond to the number of time series, the number of measurements per time series and the number of dimensions respectively (`n_ts, max_sz, d`). In order to get the data in the right format, different solutions exist:
* [You can use the utility functions such as `to_time_series_dataset`](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.utils.html#module-tslearn.utils)
* [You can convert from other popular time series toolkits in Python](https://tslearn.readthedocs.io/en/latest/integration_other_software.html)
* [You can load any of the UCR datasets in the required format.](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.datasets.html#module-tslearn.datasets)
* [You can generate synthetic data using the `generators` module](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.generators.html#module-tslearn.generators)

It should further be noted that tslearn [supports variable-length timeseries](https://tslearn.readthedocs.io/en/latest/variablelength.html).

```python3
>>> from tslearn.utils import to_time_series_dataset
>>> my_first_time_series = [1, 3, 4, 2]
>>> my_second_time_series = [1, 2, 4, 2]
>>> my_third_time_series = [1, 2, 4, 2, 2]
>>> X = to_time_series_dataset([my_first_time_series,
                                my_second_time_series,
                                my_third_time_series])
>>> y = [0, 1, 1]
```

## 2. Data preprocessing and transformations
Optionally, tslearn has several utilities to preprocess the data. In order to facilitate the convergence of different algorithms, you can [normalize time series](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.preprocessing.html#module-tslearn.preprocessing). Alternatively, in order to speed up training times, one can [resample](https://tslearn.readthedocs.io/en/latest/gen_modules/preprocessing/tslearn.preprocessing.TimeSeriesResampler.html#tslearn.preprocessing.TimeSeriesResampler) the data or apply a [piece-wise transformation](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.piecewise.html#module-tslearn.piecewise), such as SAX.

```python3
>>> from tslearn.preprocessing import TimeSeriesScalerMinMax
>>> from tslearn.piecewise import SymbolicAggregateApproximation
>>> X_norm = TimeSeriesScalerMinMax().fit_transform(X)
>>> X_sax = SymbolicAggregateApproximation(4, 10).fit_transform(X_norm)
>>> print(X_sax)
[[[5] [7] [8] [6]] 
 [[5] [6] [8] [6]]
 [[5] [6] [8] [6]]]
```

## 3. Training a model

After getting the data in the right format, a model can be trained. Depending on the use case, tslearn supports different tasks: classification, clustering and regression. For an extensive overview of possibilities, check out our [gallery of examples](https://tslearn.readthedocs.io/en/latest/auto_examples/index.html).

```python3
>>> from tslearn.shapelets import ShapeletModel
>>> clf = ShapeletModel({3: 1})
>>> clf.fit(X_sax, y)
>>> print(clf.shapelets_)
[[[4.8] [8] [6]]]
```


## 4. Evaluation and analysis

tslearn further allows to perform all different types of analysis. Examples include [calculating barycenters](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.barycenters.html#module-tslearn.barycenters) of a group of time series or calculate the distances between time series using a [variety of distance metrics](https://tslearn.readthedocs.io/en/latest/gen_modules/tslearn.metrics.html#module-tslearn.metrics).

# Documentation

The documentation, including a gallery of examples, is hosted at [readthedocs](http://tslearn.readthedocs.io/en/latest/index.html).

# Contributing

If you would like to contribute to `tslearn`, please have a look at [our contribution guidelines](CONTRIBUTING.md). A list of interesting TODO's can be found [here](https://github.com/rtavenar/tslearn/issues?utf8=✓&q=is%3Aissue%20is%3Aopen%20label%3A%22new%20feature%22%20). **If you want other ML methods for time series to be added to this TODO list, do not hesitate to open an issue!**

# Referencing tslearn

If you use `tslearn` in a scientific publication, we would appreciate citations:

```bibtex
@misc{tslearn,
 title={tslearn: A machine learning toolkit dedicated to time-series data},
 author={Tavenard, Romain and Faouzi, Johann and Vandewiele, Gilles},
 year={2017},
 note={\url{https://github.com/rtavenar/tslearn}}
}
```

### Acknowledgments
Authors would like to thank Mathieu Blondel for providing code for [Kernel k-means](https://gist.github.com/mblondel/6230787) and [Soft-DTW](https://github.com/mblondel/soft-dtw).
