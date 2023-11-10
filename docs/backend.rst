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
-------------------------

Use the backend's automatic differentiation tools in metric functions

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
    >>> sim.backward()
    >>> d_s1 = s1.grad
    >>> print(d_s1)
    tensor([[-0.3086],
            [-0.1543],
            [ 0.7715]])

.. code-block:: python

    >>> import torch
    >>> from tslearn.metrics import SoftDTWLossPyTorch
    >>> soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.1)
    >>> x = torch.zeros((4, 3, 2), requires_grad=True)
    >>> y = torch.arange(0, 24).reshape(4, 3, 2)
    >>> soft_dtw_loss_mean_value = soft_dtw_loss(x, y).mean()
    >>> print(soft_dtw_loss_mean_value)
    tensor(1081., grad_fn=<MeanBackward0>)
    >>> soft_dtw_loss_mean_value.backward()
    >>> print(x.grad.shape)
    torch.Size([4, 3, 2])
    >>> print(x.grad)
    tensor([[[  0.0000,  -0.5000],
             [ -1.0000,  -1.5000],
             [ -2.0000,  -2.5000]],

            [[ -3.0000,  -3.5000],
             [ -4.0000,  -4.5000],
             [ -5.0000,  -5.5000]],

            [[ -6.0000,  -6.5000],
             [ -7.0000,  -7.5000],
             [ -8.0000,  -8.5000]],

            [[ -9.0000,  -9.5000],
             [-10.0000, -10.5000],
             [-11.0000, -11.5000]]])
