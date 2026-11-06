"""Precision of soft-DTW loss values and first derivatives."""

import pytest

from tslearn.metrics import SoftDTWLossPyTorch

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _default_float32():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _enumerated_loss(x, y, gamma):
    """Evaluate every monotone warping path for one small time-series pair."""
    distance = (x[:, None] - y[None, :]).square().sum(-1)
    costs = []

    def visit(i, j, cost):
        cost = cost + distance[i, j]
        if i == len(x) - 1 and j == len(y) - 1:
            costs.append(cost)
            return
        for di, dj in [(1, 0), (0, 1), (1, 1)]:
            if i + di < len(x) and j + dj < len(y):
                visit(i + di, j + dj, cost)

    visit(0, 0, distance.new_zeros(()))
    return -gamma * torch.logsumexp(-torch.stack(costs) / gamma, dim=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize", [False, True])
def test_soft_dtw_precision_matches_enumerated_paths(dtype, normalize):
    """Loss and gradients retain floating-point precision."""
    x = torch.tensor([[[0.1], [0.6], [-0.3]]], dtype=dtype, requires_grad=True)
    y = torch.tensor(
        [[[0.2], [-0.5], [0.4], [0.1]]], dtype=dtype, requires_grad=True
    )
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    gamma = 0.1
    actual = SoftDTWLossPyTorch(gamma=gamma, normalize=normalize)(x, y).sum()
    expected = _enumerated_loss(x_ref[0], y_ref[0], gamma)
    if normalize:
        expected = expected - 0.5 * (
            _enumerated_loss(x_ref[0], x_ref[0], gamma)
            + _enumerated_loss(y_ref[0], y_ref[0], gamma)
        )
    actual_grads = torch.autograd.grad(actual, (x, y))
    expected_grads = torch.autograd.grad(expected, (x_ref, y_ref))
    rtol, atol = (2e-5, 2e-6) if dtype == torch.float32 else (1e-11, 1e-12)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=rtol, atol=atol
        )


def test_soft_dtw_double_precision_gradcheck():
    """Ordinary double-precision inputs pass numerical Jacobian validation."""
    x = torch.tensor(
        [[[0.1], [0.6], [-0.3]]], dtype=torch.float64, requires_grad=True
    )
    y = torch.tensor(
        [[[0.2], [-0.5], [0.4], [0.1]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(SoftDTWLossPyTorch(), (x, y))
