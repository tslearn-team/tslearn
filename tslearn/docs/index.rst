.. tslearn documentation master file, created by
   sphinx-quickstart on Mon May  8 21:34:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``tslearn``'s documentation
===========================

``tslearn`` is a Python package that provides machine learning tools for the
analysis of time series.
This package builds on (and hence depends on) ``scikit-learn``, ``numpy`` and
``scipy`` libraries.

If you plan to use the ``shapelets`` module from ``tslearn``, ``keras`` and
``tensorflow`` should also be installed.

Installation
------------

Using conda
```````````

The easiest way to install ``tslearn`` is probably via ``conda``:

.. code-block:: bash

    conda install -c conda-forge tslearn

Using PyPI
``````````

Using ``pip`` should also work fine:

.. code-block:: bash
  
    pip install tslearn


In this case, you should have ``numpy``, ``cython`` and C++ build tools
available at build time.

Using latest github-hosted version
``````````````````````````````````

If you want to get ``tslearn``'s latest version, you can refer to the
repository hosted at github:

.. code-block:: bash
  
    pip install git+https://github.com/rtavenar/tslearn.git



In this case, you should have ``numpy``, ``cython`` and C++ build tools
available at build time.


Navigation
----------

From here, you can navigate to:

.. toctree::
    :maxdepth: 1

    gettingstarted
    variablelength
    reference
    auto_examples/index

