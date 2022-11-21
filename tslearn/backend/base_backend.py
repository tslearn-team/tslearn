"""The base backend."""


class BaseBackend(object):
    """Class for the base  backend."""

    def __init__(self):
        self.linalg = BaseBackendLinalg()

    @staticmethod
    def shape(data):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def array(data, dtype=None):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def exp(data, dtype=None):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def log(data, dtype=None):
        raise NotImplementedError("Not implemented")

    @staticmethod
    def zeros(shape, dtype=None):
        raise NotImplementedError("Not implemented")


class BaseBackendLinalg:
    @staticmethod
    def inv(x):
        raise NotImplementedError("Not implemented")
