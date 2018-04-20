.. tslearn documentation master file, created by
   sphinx-quickstart on Mon May  8 21:34:49 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

``tslearn``'s documentation
===========================

``tslearn`` is a Python package that provides machine learning tools for the analysis of time series.
This package builds on (and hence depends on) ``scikit-learn``, ``numpy`` and ``scipy`` libraries.

If you plan to use the ``shapelets`` module from ``tslearn``, ``keras`` should also be installed.

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


Using latest github-hosted version
``````````````````````````````````

If you want to get ``tslearn``'s latest version, you can refer to the repository hosted at github:

.. code-block:: bash
  
    pip install git+https://github.com/rtavenar/tslearn.git


Troubleshooting
```````````````

It seems on some platforms ``Cython`` dependency does not install properly.
If you experiment such an issue, try installing it with the following command:

.. code-block:: bash

    pip install cython


or (depending on your preferred python package manager):

.. code-block:: bash

    conda install -c anaconda cython


before you start installing ``tslearn``.


Navigation
----------

From here, you can navigate to:

.. toctree::
    :maxdepth: 1

    gettingstarted
    variablelength
    reference
    auto_examples/index

