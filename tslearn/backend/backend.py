"""The generic backend."""

from tslearn.backend.numpy_backend import NumPyBackend

try:
    import torch

    from tslearn.backend.pytorch_backend import PyTorchBackend
except ImportError:

    class PyTorchBackend:
        def __init__(self):
            raise ValueError(
                "Could not use pytorch backend since torch is not installed"
            )


def instantiate_backend(*args):
    """Select backend.

    Parameter
    ---------
    *args : Input arguments can be Backend instance or string or array or None
        Arguments used to define the backend instance.

    Returns
    -------
    backend : Backend instance
        The backend instance.
    """
    for arg in args:
        if isinstance(arg, Backend):
            return arg
        if "numpy" in f"{type(arg)}" or "numpy" in f"{arg}".lower():
            return Backend("numpy")
        if "torch" in f"{type(arg)}" or "torch" in f"{arg}".lower():
            return Backend("pytorch")
    return Backend("numpy")


def select_backend(data):
    """Select backend.

    Parameter
    ---------
    data : array-like or string or None
        Indicates the backend to choose.
        Optional, default equals None.

    Returns
    -------
    backend : class
        The backend class.
        If data is a Numpy array or data equals 'numpy' or data is None,
        backend equals NumpyBackend().
        If data is a PyTorch array or data equals 'pytorch',
        backend equals PytorchBackend().
    """
    if "torch" in f"{type(data)}" or "torch" in f"{data}".lower():
        return PyTorchBackend()
    return NumPyBackend()


class Backend(object):
    """Class for the  backend.

    Parameter
    ---------
    data : array-like or string or None
        Indicates the backend to choose.
        If data is a Numpy array or data equals 'numpy' or data is None,
        self.backend is set to NumpyBackend().
        If data is a PyTorch array or data equals 'pytorch',
        self.backend is set to PytorchBackend().
        Optional, default equals None.
    """

    def __init__(self, data=None):
        self.backend = select_backend(data)

        for element in dir(self.backend):
            if "__" not in element:
                setattr(self, element, getattr(self.backend, element))

        self.is_numpy = self.backend_string == "numpy"
        self.is_pytorch = self.backend_string == "pytorch"

    def get_backend(self):
        return self.backend

    def set_backend(self, data=None):
        self.backend = select_backend(data)


def cast(data, array_type="numpy"):
    """Cast data to list or specific backend.

    Parameters
    ----------
    data: array-like,
        The input data should be a list or numpy array or torch array.
        The data to cast.
    array_type: string
        The type to cast the data. It can be "numpy", "pytorch" or "list".

    Returns
    --------
    data_cast: array-like
        Data cast to array_type.
    """
    data_type_string = f"{type(data)}".lower()
    array_type = array_type.lower()
    if array_type == "pytorch":
        array_type = "torch"
    if array_type in data_type_string:
        return data
    if array_type == "list":
        return data.tolist()
    be = Backend(array_type)
    return be.array(data)
