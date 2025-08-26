Installation
============

Using conda
-----------

The easiest way to install ``tslearn`` is probably via ``conda`` (preferably in a dedicated environment):

.. code-block:: bash

    conda install -c conda-forge tslearn

Using PyPI
----------

Using ``pip`` should also work fine (preferably in a dedicated virtual environment):

.. code-block:: bash

    python -m pip install tslearn

Using latest github-hosted version
----------------------------------

If you want to get ``tslearn``'s latest version, you can refer to the
repository hosted at github:

.. code-block:: bash

    python -m pip install https://github.com/tslearn-team/tslearn/archive/main.zip

A note on requirements
----------------------

``tslearn`` builds on (and hence depends on) ``scikit-learn``, ``numpy`` and
``scipy`` libraries. It also depends on the ``numba`` and ``joblib`` libraries.

.. include:: ./dependencies.rst

Those should automatically be pulled on a standard ``tslearn`` installation.

If you plan to use the :mod:`tslearn.shapelets` module from
``tslearn``, ``keras`` (v3+) and ``pytorch`` should also be installed.
``pytorch`` can also be used as a computational backend for some metrics.
See :doc:`the backend section <backend>` for more information.

``h5py`` is required for reading or writing models using the hdf5 file format.

The ``cesium`` and ``pandas`` libraries may also be required if you plan on
:ref:`integrating with some other python packages <integration_other_software>`.

You can use the `[all_features]` extra to enjoy all the features provided in the ``tslearn`` package:

.. code-block:: bash

    python -m pip install tslearn[all_features]
