Installation
============

Using conda
-----------

The easiest way to install ``tslearn`` is probably via ``conda``:

.. code-block:: bash

    conda install -c conda-forge tslearn

Using PyPI
----------

Using ``pip`` should also work fine:

.. code-block:: bash

    python -m pip install tslearn

In this case, you should have ``numpy``, ``cython`` and C++ build tools
available at build time.


Using latest github-hosted version
----------------------------------

If you want to get ``tslearn``'s latest version, you can refer to the
repository hosted at github:

.. code-block:: bash

    python -m pip install https://github.com/tslearn-team/tslearn/archive/master.zip

In this case, you should have ``numpy``, ``cython`` and C++ build tools
available at build time.


It seems on some platforms ``Cython`` dependency does not install properly.
If you experiment such an issue, try installing it with the following command:

.. code-block:: bash

    python -m pip install cython


before you start installing ``tslearn``.
If it still does not work, we suggest you switch to `conda` installation.

Other requirements
------------------

``tslearn`` builds on (and hence depends on) ``scikit-learn``, ``numpy`` and
``scipy`` libraries.

If you plan to use the :mod:`tslearn.shapelets` module from
``tslearn``, ``tensorflow`` (v2) should also be installed.
``h5py`` is required for reading or writing models using the hdf5 file format.
In order to load multivariate datasets from the UCR/UEA archive using the
:class:`tslearn.datasets.UCR_UEA_datasets` class,
installed ``scipy`` version should be greater than 1.3.0.

