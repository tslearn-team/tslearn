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


def test_soft_dtw_loss_pytorch_precision():
    torch = pytest.importorskip("torch")

    for dtype, eps in [(torch.float64, 1e-06), (torch.float32, 1e-03)]:
        x = torch.tensor([[[0.1], [0.6], [-0.3]]], dtype=dtype, requires_grad=True)
        y = torch.tensor([[[0.2], [-0.5], [0.4], [0.1]]], dtype=dtype, requires_grad=True)
        assert torch.autograd.gradcheck(tslearn.metrics.SoftDTWLossPyTorch(), (x, y), eps=eps)
