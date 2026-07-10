"""The Soft-DTW loss function for PyTorch.

The Soft-DTW loss function for PyTorch proposed is inspired by Maghoumi's GitHub repository:
https://github.com/Maghoumi/pytorch-softdtw-cuda/blob/master/soft_dtw_cuda.py
"""

import numpy as np

from numba import njit, prange

try:
    import torch

except ImportError:
    torch = None


from tslearn.metrics import GLOBAL_CONSTRAINT_CODE
from tslearn.metrics._masks import _njit_compute_mask
from tslearn.metrics._soft_dtw import _njit_accumulated_matrix_from_distance_matrix, _njit_soft_dtw_grad


if torch is None:

    class SoftDTWLossPyTorch:
        def __init__(self):
            raise ValueError(
                "Could not use SoftDTWLossPyTorch since torch is not installed"
            )

else:

    class _SoftDTWLossPyTorch(torch.autograd.Function):
        """The Soft-DTW loss function."""

        @staticmethod
        def forward(
            ctx,
            D,
            gamma,
            global_constraint=None,
            sakoe_chiba_radius=None,
            itakura_max_slope=None
        ):
            r"""
            Parameters
            ----------
            ctx : context
            D : Tensor, shape=[b, m, n]
                Matrix of pairwise distances.
            gamma : float
                Regularization parameter.
                Lower is less smoothed (closer to true DTW).
            global_constraint : {0, 1, 2} (default: 0)
                Global constraint to restrict admissible paths for DTW:
                - "itakura" if 1
                - "sakoe_chiba" if 2
                - no constraint otherwise
            sakoe_chiba_radius : int or None (default: None)
                Radius to be used for Sakoe-Chiba band global constraint.
                The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
                it controls how far in time we can go in order to match a given
                point from one time series to a point in another time series.
                If None and `global_constraint` is set to 2 (sakoe-chiba), a radius of
                1 is used.
                If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
                `global_constraint` is used to infer which constraint to use among the
                two. In this case, if `global_constraint` corresponds to no global
                constraint, a `RuntimeWarning` is raised and no global constraint is
                used.
            itakura_max_slope : float or None (default: None)
                Maximum slope for the Itakura parallelogram constraint.
                If None and `global_constraint` is set to 1 (itakura), a maximum slope
                of 2. is used.
                If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
                `global_constraint` is used to infer which constraint to use among the
                two. In this case, if `global_constraint` corresponds to no global
                constraint, a `RuntimeWarning` is raised and no global constraint is
                used.

            Returns
            -------
            loss : Tensor, shape=[batch_size,]
                The loss values.
            """
            dev = D.device
            dtype = D.dtype
            _, m, n = D.shape
            mask = _njit_compute_mask(m, n, global_constraint, sakoe_chiba_radius, itakura_max_slope)
            D[:, ~mask] = torch.inf
            D_ = D.detach().cpu().numpy()
            R_ = _njit_soft_dtw_batch(D_, gamma, mask)
            gamma_tensor = torch.Tensor([gamma]).to(dev).type(dtype)
            R = torch.Tensor(R_).to(dev).type(dtype)
            ctx.save_for_backward(D, R,gamma_tensor)
            return R[:, -1, -1]

        @staticmethod
        def backward(ctx, grad_output):
            """Backward pass."""
            dev = grad_output.device
            dtype = grad_output.dtype
            D, R, gamma_tensor = ctx.saved_tensors
            D_ = D.detach().cpu().numpy()
            R_ = R.detach().cpu().numpy()
            gamma = gamma_tensor.item()
            E_ = _njit_soft_dtw_grad_batch(D_, R_, gamma)
            E = torch.Tensor(E_).to(dev).type(dtype)
            return grad_output.reshape(-1).to(dtype=E.dtype).view(-1, 1, 1) * E, None, None, None, None


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
        global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
            Global constraint to restrict admissible paths for DTW.
        sakoe_chiba_radius : int or None (default: None)
            Radius to be used for Sakoe-Chiba band global constraint.
            The Sakoe-Chiba radius corresponds to the parameter :math:`\delta` mentioned in [1]_,
            it controls how far in time we can go in order to match a given
            point from one time series to a point in another time series.
            If None and `global_constraint` is set to "sakoe_chiba", a radius of
            1 is used.
            If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
            `global_constraint` is used to infer which constraint to use among the
            two. In this case, if `global_constraint` corresponds to no global
            constraint, a `RuntimeWarning` is raised and no global constraint is
            used.
        itakura_max_slope : float or None (default: None)
            Maximum slope for the Itakura parallelogram constraint.
            If None and `global_constraint` is set to "itakura", a maximum slope
            of 2. is used.
            If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
            `global_constraint` is used to infer which constraint to use among the
            two. In this case, if `global_constraint` corresponds to no global
            constraint, a `RuntimeWarning` is raised and no global constraint is
            used.

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
                 [ -2.0000,  -2.5001]],
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

        def __init__(
            self,
            gamma=1.0,
            normalize=False,
            dist_func=None,
            global_constraint=None,
            sakoe_chiba_radius=None,
            itakura_max_slope=None,
        ):
            super().__init__()
            self.gamma = gamma
            self.normalize = normalize
            if dist_func is not None:
                self.dist_func = dist_func
            else:
                self.dist_func = self._euclidean_squared_dist
            self.global_constraint = global_constraint
            self.sakoe_chiba_radius = sakoe_chiba_radius
            self.itakura_max_slope = itakura_max_slope

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
            loss_xy = _SoftDTWLossPyTorch.apply(
                d_xy,
                self.gamma,
                GLOBAL_CONSTRAINT_CODE[self.global_constraint],
                self.sakoe_chiba_radius,
                self.itakura_max_slope
            )
            if self.normalize:
                d_xx = self.dist_func(x, x)
                d_yy = self.dist_func(y, y)
                loss_xx = _SoftDTWLossPyTorch.apply(
                    d_xx,
                    self.gamma,
                    GLOBAL_CONSTRAINT_CODE[self.global_constraint],
                    self.sakoe_chiba_radius,
                    self.itakura_max_slope
                )
                loss_yy = _SoftDTWLossPyTorch.apply(
                    d_yy,
                    self.gamma,
                    GLOBAL_CONSTRAINT_CODE[self.global_constraint],
                    self.sakoe_chiba_radius,
                    self.itakura_max_slope
                )
                return loss_xy - 1 / 2 * (loss_xx + loss_yy)
            else:
                return loss_xy


@njit(parallel=True)
def _njit_soft_dtw_batch(
    D,
    gamma,
    mask
):
    b, m, n = D.shape
    R = np.empty((b, m, n),dtype=D.dtype)
    for i in prange(b):
        _njit_accumulated_matrix_from_distance_matrix(D[i], mask, gamma, out=R[i])
    return R


@njit(parallel=True)
def _njit_soft_dtw_grad_batch(D, R, gamma):
    b, m, n = D.shape
    E = np.zeros((b, m, n), dtype=D.dtype)
    for i in prange(b):
        _njit_soft_dtw_grad(D[i], R[i], gamma, out=E[i])
    return E
