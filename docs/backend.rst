Backend selection and use
=========================

`tslearn` proposes time series metrics as `DTW` and `Soft-DTW` which are compatible with different backends (`NumPy` and `PyTorch`).

Backend selection
-----------------

A backend can be initialized passing four different kind of input arguments to the function instantiate_backend:

* Input is a Backend instance
* Input is a string corresponding to "numpy" or "pytorch".
* Input is a backend object.
* Input is "None" or anything else than mentioned previously --> The backend `NumPy` is used. 

What happens if there are several inputs? --> A for loop on the inputs.

Examples
~~~~~~~~

.. code-block:: python

    >>> from tslearn.backend import instantiate_backend
    >>> be = instantiate_backend("pytorch")
    >>> print(be.backend_string)
    "pytorch"

.. code-block:: python

    >>> from tslearn.backend import Backend
    
.. code-block:: python

    >>> from tslearn.backend import Backend

Create backend objects
----------------------

`NumPy` is the reference backend concerning the names of the methods of the backends.

Examples
~~~~~~~~

.. code-block:: python

    >>> from tslearn.backend import Backend
    >>> be = instantiate_backend("pytorch")
    >>> print(be.array([0]))
    Torch.Tensor([0])

Define metric functions backend
-------------------------------

Pass a backend as an optional input argument of a metric function.


Examples
~~~~~~~~

.. code-block:: python

    >>> from tslearn.backend import Backend

Automatic differentiation
----------

Use the backend's automatic differentiation tools in metric functions

Examples
~~~~~~~~

.. code-block:: python

    >>> from tslearn.backend import Backend

.. code-block:: python

    >>> from tslearn.backend import Backend

.. code-block:: python

    >>> from tslearn.backend import Backend

