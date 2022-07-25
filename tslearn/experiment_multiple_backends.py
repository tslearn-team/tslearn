import numpy as np
import torch


X_numpy = np.array([1.0, 2.0, -1.0])
X_torch = torch.tensor([1.0, 2.0, 3.0])

print(X_numpy)
print(X_torch)


class Backend(object):

    def __init__(self):
        pass

    def array(self, data):
        return NotImplementedError('Not implemented')


class NumpyBackend(Backend):

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype)


numpy_backend = NumpyBackend()
print(numpy_backend.array([5, 6, 4.0]))


class PyTorchBackend(Backend):

    def array(self, data, dtype=torch.float32):
        return torch.tensor(data, dtype=dtype)


class GenericBackend(object):

    def __init__(self):
        self.backend = Backend()

    def get_backend(self, data):
        if 'numpy' in f"{type(data)}":
            self.backend = NumpyBackend()
            return NumpyBackend()
        if 'torch' in f"{type(data)}":
            self.backend = PyTorchBackend()
            return PyTorchBackend()

    def array(self, data):
        return self.backend.array(data)


class TestClassifier(object):
    def __init__(self):
        self.backend = GenericBackend()

    def fit(self, X, y=None):
        self.backend = GenericBackend().get_backend(X)
        return X

    def predict(self, X):
        return self.backend.array([1.0, 2.0, 3.0, 4.0])


testclassifier = TestClassifier()
testclassifier.fit(X_numpy)
predict_numpy = testclassifier.predict(X_numpy)
print(predict_numpy)
print(type(predict_numpy))

testclassifier.fit(X_torch)
predict_torch = testclassifier.predict(X_torch)
print(predict_torch)
print(type(predict_torch))
