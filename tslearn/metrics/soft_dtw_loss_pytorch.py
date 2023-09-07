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
        r"""Soft-DTW loss function in PyTorch.

        Soft-DTW was originally presented in [1]_ and is
        discussed in more details in our
        :ref:`user-guide page on DTW and its variants<dtw>`.

        Soft-DTW is computed as:

        .. math::

            \text{soft-DTW}_{\gamma}(X, Y) =
                \min_{\pi}{}^\gamma \sum_{(i, j) \in \pi} d \left( X_i, Y_j \right)

        where :math:`d` is a distance function or a dissimilarity measure
        supporting PyTorch automatic differentiation and :math:`\min^\gamma` is
        the soft-min operator of parameter :math:`\gamma` defined as:

        .. math::

            \min{}^\gamma \left( a_{1}, ..., a_{n} \right) =
                - \gamma \log \sum_{i=1}^{n} e^{- a_{i} / \gamma}

        In the limit case :math:`\gamma = 0`, :math:`\min^\gamma` reduces to a
        hard-min operator. The soft-DTW is then defined as the square of the DTW
        dissimilarity measure when :math:`d` is the squared Euclidean distance.

        Contrary to DTW, soft-DTW is not bounded below by zero, and we even have:

        .. math::

            \text{soft-DTW}_{\gamma}(X, Y) \rightarrow - \infty \text{ when } \gamma \rightarrow + \infty

        In [2]_, new dissimilarity measures are defined, that rely on soft-DTW.
        In particular, soft-DTW divergence is introduced to counteract the non-positivity of soft-DTW:

        .. math::
            D_{\gamma} \left( X, Y \right) =
                \text{soft-DTW}_{\gamma}(X, Y)
                - \frac{1}{2} \left( \text{soft-DTW}_{\gamma}(X, X) + \text{soft-DTW}_{\gamma}(Y, Y) \right)

        This divergence has the advantage of being minimized for :math:`X = Y`
        and being exactly 0 in that case.

        Parameters
        ----------
        gamma : float
            Regularization parameter.
            It should be strictly positive.
            Lower is less smoothed (closer to true DTW).
        normalize : bool
            If True, the Soft-DTW divergence is used.
            The Soft-DTW divergence is always positive.
            Optional, default: False.
        dist_func : callable
            Distance function or dissimilarity measure.
            It takes two input arguments of shape (batch_size, ts_length, dim).
            It should support PyTorch automatic differentiation.
            Optional, default: None
            If None, the squared Euclidean distance is used.

        Examples
        --------
        >>> import torch
        >>> from tslearn.metrics import SoftDTWLossPyTorch
        >>> soft_dtw_loss = SoftDTWLossPyTorch(gamma=0.1)
        >>> x = torch.zeros((4, 3, 2), requires_grad=True)
        >>> y = torch.arange(0, 24).reshape(4, 3, 2)
        >>> soft_dtw_loss_mean_value = soft_dtw_loss(x, y).mean()
        >>> print(soft_dtw_loss_mean_value)
        tensor(1081., grad_fn=<MeanBackward0>)
        >>> soft_dtw_loss_mean_value.backward()
        >>> print(x.grad.shape)
        torch.Size([4, 3, 2])
        >>> print(x.grad)
        tensor([[[  0.0000,  -0.5000],
                 [ -1.0000,  -1.5000],
                 [ -2.0000,  -2.5000]],
        <BLANKLINE>
                [[ -3.0000,  -3.5000],
                 [ -4.0000,  -4.5000],
                 [ -5.0000,  -5.5000]],
        <BLANKLINE>
                [[ -6.0000,  -6.5000],
                 [ -7.0000,  -7.5000],
                 [ -8.0000,  -8.5000]],
        <BLANKLINE>
                [[ -9.0000,  -9.5000],
                 [-10.0000, -10.5000],
                 [-11.0000, -11.5000]]])

        See Also
        --------
        soft_dtw : Compute Soft-DTW metric between two time series.
        cdist_soft_dtw : Compute cross-similarity matrix using Soft-DTW metric.
        cdist_soft_dtw_normalized : Compute cross-similarity matrix using a normalized
            version of the Soft-DTW metric.

        References
        ----------
        .. [1] Marco Cuturi & Mathieu Blondel. "Soft-DTW: a Differentiable Loss Function for
           Time-Series", ICML 2017.
        .. [2] Mathieu Blondel, Arthur Mensch & Jean-Philippe Vert.
            "Differentiable divergences between time series",
            International Conference on Artificial Intelligence and Statistics, 2021.
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
            d_xy = self.dist_func(x, y)
            loss_xy = _SoftDTWLossPyTorch.apply(d_xy, self.gamma)
            if self.normalize:
                d_xx = self.dist_func(x, x)
                d_yy = self.dist_func(y, y)
                loss_xx = _SoftDTWLossPyTorch.apply(d_xx, self.gamma)
                loss_yy = _SoftDTWLossPyTorch.apply(d_yy, self.gamma)
                return loss_xy - 1 / 2 * (loss_xx + loss_yy)
            else:
                return loss_xy
