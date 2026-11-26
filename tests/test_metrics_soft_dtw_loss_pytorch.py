import functools

import pytest

import tslearn


@pytest.mark.skipif(tslearn.backend.pytorch_backend.HAS_TORCH, reason="PyTorch is installed")
def test_no_torch():
    with pytest.raises(ValueError,
                       match="Could not use SoftDTWLossPyTorch since torch is not installed"):
        tslearn.metrics.SoftDTWLossPyTorch()


def test_soft_dtw_loss_pytorch():
    """Tests for the class SoftDTWLossPyTorch."""
    torch = pytest.importorskip('torch')

    b, m, n, d = 5, 10, 12, 8
    batch_ts_1 = torch.zeros((b, m, d), requires_grad=True)
    batch_ts_2 = torch.ones((b, n, d), requires_grad=True)

    soft_dtw_loss_pytorch = tslearn.metrics.SoftDTWLossPyTorch()
    loss = soft_dtw_loss_pytorch(batch_ts_1, batch_ts_2)
    torch.testing.assert_close(loss, 91.9806137 * torch.ones(b))
    loss = soft_dtw_loss_pytorch(batch_ts_1, batch_ts_1)
    torch.testing.assert_close(loss, -14.1957006 * torch.ones(b))

    soft_dtw_loss_pytorch_normalized = tslearn.metrics.SoftDTWLossPyTorch(normalize=True)
    loss_normalized = soft_dtw_loss_pytorch_normalized(batch_ts_1, batch_ts_1)
    torch.testing.assert_close(loss_normalized, torch.zeros(b))
    loss_normalized = soft_dtw_loss_pytorch_normalized(batch_ts_2, batch_ts_2)
    torch.testing.assert_close(loss_normalized, torch.zeros(b))
    loss_normalized = soft_dtw_loss_pytorch_normalized(batch_ts_1, batch_ts_2)
    torch.testing.assert_close(loss_normalized, 107.89006805 * torch.ones((b,)))

    custom_dist = functools.partial(torch.cdist, p=1)
    soft_dtw_loss_pytorch = tslearn.metrics.SoftDTWLossPyTorch(dist_func=custom_dist)
    loss = soft_dtw_loss_pytorch(batch_ts_1, batch_ts_2)
    torch.testing.assert_close(loss, 91.9806137 * torch.ones(b))

    soft_dtw_loss_pytorch.gamma=0.1
    loss = soft_dtw_loss_pytorch(batch_ts_1, batch_ts_2)
    loss.mean().backward()
    expected_grad_ts1 = torch.full((b, m, d), -0.24)
    torch.testing.assert_close(batch_ts_1.grad, expected_grad_ts1)
    expected_grad_ts2 = torch.full((b, n, d), 0.20)
    torch.testing.assert_close(batch_ts_2.grad, expected_grad_ts2)

    batch_ts_1 = torch.zeros((b, m, d), requires_grad=True)
    batch_ts_2 = torch.ones((b, n, d), requires_grad=True)
    soft_dtw_loss_pytorch = tslearn.metrics.SoftDTWLossPyTorch(
        gamma=0.1,
        global_constraint="sakoe_chiba",
        sakoe_chiba_radius=1
    )
    loss = soft_dtw_loss_pytorch(batch_ts_1, batch_ts_2)
    loss.mean().backward()
    expected_grad_ts1 = torch.full((b, m, d), -0.48)
    torch.testing.assert_close(batch_ts_1.grad, expected_grad_ts1)
    expected_grad_ts2 = torch.full((b, n, d), 0.40)
    torch.testing.assert_close(batch_ts_2.grad, expected_grad_ts2)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_soft_dtw_padded_pairs(normalize, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x_lengths, y_lengths = [3, 1, 3, 4], [2, 3, 2, 4]
    x, y = (torch.arange(32, device=device, dtype=torch.float64).reshape(4, 4, 2) / scale
            for scale in (15, 23))
    for i, (m, n) in enumerate(zip(x_lengths, y_lengths)):
        x[i, m:] = torch.nan
        y[i, n:] = torch.nan
    x.requires_grad_()
    y.requires_grad_()
    loss = tslearn.metrics.SoftDTWLossPyTorch(
        normalize=normalize, dist_func=functools.partial(torch.cdist, p=1),
        global_constraint="sakoe_chiba", sakoe_chiba_radius=1,
    )
    actual = loss(x, y, x_lengths=x_lengths, y_lengths=torch.tensor(y_lengths, device=device))
    expected = torch.cat([loss(x[i:i+1, :m], y[i:i+1, :n])
                          for i, (m, n) in enumerate(zip(x_lengths, y_lengths))])
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(1, 5, device=device, dtype=x.dtype)
    for grad, reference in zip(torch.autograd.grad(actual @ weights, (x, y)),
                               torch.autograd.grad(expected @ weights, (x, y))):
        torch.testing.assert_close(grad, reference)


def test_soft_dtw_lengths():
    torch = pytest.importorskip("torch")
    x = torch.ones(2, 4, 1)
    loss = tslearn.metrics.SoftDTWLossPyTorch()
    torch.testing.assert_close(loss(x, x, y_lengths=[2, 2]), loss(x, x[:, :2]))
    for lengths in ([0, 2], [1, 5], [2], [[1, 2]], [1., 2.], [True, False]):
        with pytest.raises(ValueError):
            loss(x, x, x_lengths=lengths)
    empty = x[:0].requires_grad_()
    values = loss(empty, empty, x_lengths=[], y_lengths=[])
    assert values.shape == (0,)
    values.sum().backward()
    torch.testing.assert_close(empty.grad, torch.zeros_like(empty))


def test_soft_dtw_loss_pytorch_precision():
    torch = pytest.importorskip("torch")

    for dtype, eps in [(torch.float64, 1e-06), (torch.float32, 1e-03)]:
        x = torch.tensor([[[0.1], [0.6], [-0.3]]], dtype=dtype, requires_grad=True)
        y = torch.tensor([[[0.2], [-0.5], [0.4], [0.1]]], dtype=dtype, requires_grad=True)
        assert torch.autograd.gradcheck(tslearn.metrics.SoftDTWLossPyTorch(), (x, y), eps=eps)
