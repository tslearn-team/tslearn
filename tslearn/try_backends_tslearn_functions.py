from try_backends_definition import GenericBackend

ts = GenericBackend()


def add(x, y):
    return x + y


def exp(x):
    ts.get_backend(x)
    return ts.exp(x)


def log(x):
    ts.get_backend(x)
    return ts.log(x)
