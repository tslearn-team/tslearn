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

    python -m pip install https://github.com/rtavenar/tslearn/archive/master.zip

In this case, you should have ``numpy``, ``cython`` and C++ build tools
available at build time.


It seems on some platforms ``Cython`` dependency does not install properly.
If you experiment such an issue, try installing it with the following command:

.. code-block:: bash

    python -m pip install cython


before you start installing ``tslearn``.
If it still does not work, we suggest you switch to `conda` installation.

