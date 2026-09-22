import pytest

def test_import():
    import tslearn
    assert "__version__" in dir(tslearn)

    with pytest.raises(AttributeError):
        tslearn.invalid
