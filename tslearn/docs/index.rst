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

Using PyPI
``````````

The easiest way to install ``tslearn`` is probably via ``pip``:

.. code-block:: bash
  
    pip install tslearn


Using latest github-hosted version
``````````````````````````````````

If you want to get ``tslearn``'s latest version, you can ``git clone`` the repository hosted at github:

.. code-block:: bash
  
    git clone https://github.com/rtavenar/tslearn.git

Then, you should run the following command for Cython code to compile:

.. code-block:: bash
  
    python setup.py build_ext --inplace

Also, for the whole package to run properly, its base directory should be appended to your Python path (you do not have
to care about it if you used the PyPI installation alternative).

Navigation
----------

From here, you can navigate to:

.. toctree::
    :maxdepth: 1

    gettingstarted
    reference
    auto_examples/index

