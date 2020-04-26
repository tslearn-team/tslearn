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
``h5py`` is required for reading or writing models using the hdf5 file format.

This documentation contains :doc:`a quick-start guide <quickstart>` (including
:doc:`installation procedure <installation>` and
:doc:`basic usage of the toolkit <gettingstarted>`),
:doc:`a complete API Reference <reference>`, as well as a
:doc:`gallery of examples <auto_examples/index>`.

Finally, if you use ``tslearn`` in a scientific publication,
:doc:`we would appreciate citations <citing>`.


.. toctree::
    :hidden:
    :maxdepth: 2

    quickstart
    reference
    auto_examples/index
    citing
