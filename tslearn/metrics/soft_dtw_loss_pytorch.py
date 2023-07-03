"""The Soft-DTW loss function for PyTorch.

The Soft-DTW loss function for PyTorch proposed is inspired by Maghoumi's GitHub repository:
https://github.com/Maghoumi/pytorch-softdtw-cuda/blob/master/soft_dtw_cuda.py
"""

import numpy as np

from .soft_dtw_fast import _njit_soft_dtw_batch, _njit_soft_dtw_grad_batch

try:
    import torch
    from torch.autograd import Function

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if not HAS_TORCH:

    class SoftDTWLossPyTorch:
        def __init__(self):
            raise ValueError(
                "Could not use SoftDTWLossPyTorch since torch is not installed"
            )

else:

    class _SoftDTWLossPyTorch(Function):
        """The Soft-DTW loss function."""

        @staticmethod
        def forward(ctx, D, gamma):
            """
            Parameters
            ----------
            ctx : context
            D : Tensor, shape=[b, m, n]
                Matrix of pairwise distances.
            gamma : float
                Regularization parameter.
                Lower is less smoothed (closer to true DTW).

            Returns
            -------
            loss : Tensor, shape=[batch_size,]
                The loss values.
            """
            dev = D.device
            dtype = D.dtype
            b, m, n = torch.Tensor.size(D)
            D_ = D.detach().cpu().numpy()
            R_ = np.zeros((b, m + 2, n + 2), dtype=np.float64)
            _njit_soft_dtw_batch(D_, R_, gamma)
            gamma_tensor = torch.Tensor([gamma]).to(dev).type(dtype)
            R = torch.Tensor(R_).to(dev).type(dtype)
            ctx.save_for_backward(D, R, gamma_tensor)
            return R[:, -2, -2]

        @staticmethod
        def backward(ctx, grad_output):
            dev = grad_output.device
            dtype = grad_output.dtype
            D, R, gamma_tensor = ctx.saved_tensors
            b, m, n = torch.Tensor.size(D)
            D_ = D.detach().cpu().numpy()
            R_ = R.detach().cpu().numpy()
            E_ = np.zeros((b, m + 2, n + 2), dtype=np.float64)
            gamma = gamma_tensor.item()
            _njit_soft_dtw_grad_batch(D_, R_, E_, gamma)
            E = torch.Tensor(E_[:, 1 : m + 1, 1 : n + 1]).to(dev).type(dtype)
            return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

    class SoftDTWLossPyTorch(torch.nn.Module):
        """Class for the Soft-DTW loss function in PyTorch.

        Parameters
        ----------
        gamma : float
            Regularization parameter.
            Lower is less smoothed (closer to true DTW).
        normalize : bool
            If True, the Soft-DTW divergence is used.
            The Soft-DTW divergence is always positive.
            Optional, default: False.
        dist_func : callable
            The distance function.
            It takes two input arguments of shape (batch_size, ts_length, dim).
            It should support PyTorch automatic differentiation.
            Optional, default: None
        """

        def __init__(self, gamma=1.0, normalize=False, dist_func=None):
            super(SoftDTWLossPyTorch, self).__init__()
            self.gamma = gamma
            self.normalize = normalize
            if dist_func is not None:
                self.dist_func = dist_func
            else:
                self.dist_func = SoftDTWLossPyTorch._euclidean_squared_dist

        @staticmethod
        def _euclidean_squared_dist(x, y):
            """Calculates the Euclidean squared distance between each element in x and y per timestep.

            Parameters
            ----------
            x : Tensor, shape=[b, m, d]
                Batch of time series.
            y : Tensor, shape=[b, n, d]
                Batch of time series.

            Returns
            -------
            dist : Tensor, shape=[b, m, n]
                The pairwise squared Euclidean distances.
            """
            m = x.size(1)
            n = y.size(1)
            d = x.size(2)
            x = x.unsqueeze(2).expand(-1, m, n, d)
            y = y.unsqueeze(1).expand(-1, m, n, d)
            return torch.pow(x - y, 2).sum(3)

        def forward(self, x, y):
            """Compute the soft-DTW value between X and Y.

            Parameters
            ----------
            x : Tensor, shape=[batch_size, ts_length, dim]
                Batch of time series.
            y : Tensor, shape=[batch_size, ts_length, dim]
                Batch of time series.

            Returns
            -------
            loss : Tensor, shape=[batch_size,]
                The loss values.
            """
            bx, lx, dx = x.shape
            by, ly, dy = y.shape
            assert bx == by
            assert dx == dy
            if self.normalize:
                xxy = torch.cat([x, x, y])
                yxy = torch.cat([y, x, y])
                d_xxy_yxy = self.dist_func(xxy, yxy)
                loss = _SoftDTWLossPyTorch.apply(d_xxy_yxy, self.gamma)
                loss_xy, loss_xx, loss_yy = torch.split(loss, x.shape[0])
                return loss_xy - 1 / 2 * (loss_xx + loss_yy)
            else:
                d_xy = self.dist_func(x, y)
                return _SoftDTWLossPyTorch.apply(d_xy, self.gamma)
