Backend selection and use
=========================

`tslearn` proposes different backends (`NumPy` and `PyTorch`)
to compute time series metrics such as `DTW` and `Soft-DTW`.
The `PyTorch` backend can be used to compute gradients of
metric functions thanks to automatic differentiation.

Backend selection
-----------------

A backend can be instantiated using the function ``instantiate_backend``.
To specify which backend should be instantiated (`NumPy` or `PyTorch`),
this function accepts four different kind of input parameters:

* a string equal to ``"numpy"`` or ``"pytorch"``.
* a `NumPy` array or a `Torch` tensor.
* a Backend instance. The input backend is then returned.
* ``None`` or anything else than mentioned previously. The backend `NumPy` is then instantiated.

Examples
~~~~~~~~

If the input is the string ``"numpy"``, the ``NumPyBackend`` is instantiated.

.. code-block:: python

    >>> from tslearn.backend import instantiate_backend
    >>> be = instantiate_backend("numpy")
    >>> print(be.backend_string)
    "numpy"

If the input is the string ``"pytorch"``, the ``PyTorchBackend`` is instantiated.

.. code-block:: python

    >>> be = instantiate_backend("pytorch")
    >>> print(be.backend_string)
    "pytorch"

If the input is a `NumPy` array, the ``NumPyBackend`` is instantiated.

.. code-block:: python

    >>> import numpy as np
    >>> be = instantiate_backend(np.array([0]))
    >>> print(be.backend_string)
    "numpy"

If the input is a `Torch` tensor, the ``PyTorchBackend`` is instantiated.

.. code-block:: python

    >>> import torch
    >>> be = instantiate_backend(torch.tensor([0]))
    >>> print(be.backend_string)
    "pytorch"

If the input is a Backend instance, the input backend is returned.

.. code-block:: python

    >>> print(be.backend_string)
    "pytorch"
    >>> be = instantiate_backend(be)
    >>> print(be.backend_string)
    "pytorch"

If the input is ``None``, the ``NumPyBackend`` is instantiated.

.. code-block:: python

    >>> be = instantiate_backend(None)
    >>> print(be.backend_string)
    "numpy"

If the input is anything else, the ``NumPyBackend`` is instantiated.

.. code-block:: python

    >>> be = instantiate_backend("Hello, World!")
    >>> print(be.backend_string)
    "numpy"

The function ``instantiate_backend`` accepts any number of input parameters, including zero.
To select which backend should be instantiated (`NumPy` or `PyTorch`),
a for loop is performed on the inputs until a backend is selected.

.. code-block:: python

    >>> be = instantiate_backend(1, None, "Hello, World!", torch.tensor([0]), "numpy")
    >>> print(be.backend_string)
    "pytorch"

If none of the inputs are related to `NumPy` or `PyTorch`, the ``NumPyBackend`` is instantiated.

.. code-block:: python

    >>> be = instantiate_backend(1, None, "Hello, World!")
    >>> print(be.backend_string)
    "numpy"

Use the backends
----------------

The names of the attributes and methods of the backends
are inspired by the `NumPy` backend.

Examples
~~~~~~~~

Create backend objects.

.. code-block:: python

    >>> be = instantiate_backend("pytorch")
    >>> mat = be.array([[0 , 1], [2, 3]], dtype=float)
    >>> print(mat)
    tensor([[0., 1.],
            [2., 3.]], dtype=torch.float64)

Use backend functions.

.. code-block:: python

    >>> norm = be.linalg.norm(mat)
    >>> print(norm)
    tensor(3.7417, dtype=torch.float64)

Choose the backend used by metric functions
-------------------------------------------

`tslearn`'s metric functions have an optional input parameter "``be``" to specify the
backend to use to compute the metric.

Examples
~~~~~~~~

.. code-block:: python

    >>> import torch
    >>> from tslearn.metrics import dtw
    >>> s1 = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    >>> s2 = torch.tensor([[3.0], [4.0], [-3.0]])
    >>> sim = dtw(s1, s2, be="pytorch")
    >>> print(sim)
    sim tensor(6.4807, grad_fn=<SqrtBackward0>)

By default, the optional input parameter ``be`` is equal to ``None``.
Note that the first line of the function ``dtw`` is:

.. code-block:: python

    be = instantiate_backend(be, s1, s2)

Therefore, even if ``be=None``, the ``PyTorchBackend`` is instantiated and used to compute the
DTW metric since ``s1`` and ``s2`` are `Torch` tensors.

.. code-block:: python

    >>> sim = dtw(s1, s2)
    >>> print(sim)
    sim tensor(6.4807, grad_fn=<SqrtBackward0>)

Automatic differentiation
-------------------------

The `PyTorch` backend can be used to compute the gradients of the metric functions thanks to automatic differentiation.

Examples
~~~~~~~~

Compute the gradient of the Dynamic Time Warping similarity measure.

.. code-block:: python

    >>> s1 = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    >>> s2 = torch.tensor([[3.0], [4.0], [-3.0]])
    >>> sim = dtw(s1, s2, be="pytorch")
    >>> sim.backward()
    >>> d_s1 = s1.grad
    >>> print(d_s1)
    tensor([[-0.3086],
            [-0.1543],
            [ 0.7715]])

Compute the gradient of the Soft-DTW similarity measure.

.. code-block:: python

    >>> from tslearn.metrics import soft_dtw
    >>> ts1 = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    >>> ts2 = torch.tensor([[3.0], [4.0], [-3.0]])
    >>> sim = soft_dtw(ts1, ts2, gamma=1.0, be="pytorch", compute_with_backend=True)
    >>> print(sim)
    tensor(41.1876, dtype=torch.float64, grad_fn=<SelectBackward0>)
    >>> sim.backward()
    >>> d_ts1 = ts1.grad
    >>> print(d_ts1)
    tensor([[-4.0001],
            [-2.2852],
            [10.1643]])
