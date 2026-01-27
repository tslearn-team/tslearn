import pytest

from tslearn.backend import Backend
from tslearn.backend.pytorch_backend import HAS_TORCH, PyTorchBackend
from tslearn.backend.numpy_backend import NumPyBackend


@pytest.mark.skipif(HAS_TORCH, reason="PyTorch is installed")
def test_backend_no_torch():
    with pytest.raises(ValueError, match="Could not use the PyTorch backend"):
        Backend("torch")


def test_backend():
    backend_ = Backend("torch")
    assert backend_.is_pytorch
    assert not backend_.is_numpy
    assert isinstance(backend_.get_backend(), PyTorchBackend)

    backend_.set_backend("numpy")
    assert not backend_.is_pytorch
    assert backend_.is_numpy
    assert isinstance(backend_.get_backend(), NumPyBackend)
