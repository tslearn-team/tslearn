import math

from numba import cuda

import torch

from tslearn.metrics._masks import _compute_mask


def __make_diag_cuda_acc_matrix_from_dist_matrix(acc_fun):

    @cuda.jit
    def _cuda_diag_acc_matrix_from_dist_matrix(D, R, mask, diag_index, args=None):
        block_index = cuda.blockIdx.x
        thread_index = cuda.threadIdx.x
        block_width = cuda.blockDim.x

        pos = thread_index + block_index * block_width

        m, n = D.shape
        i_min = max(0, diag_index - n + 1)
        i_max = min(m - 1, diag_index)
        diag_length = i_max - i_min + 1

        if pos >= diag_length:
            return

        i = i_min + pos
        j = diag_index - i

        if not mask[i, j]:
            return

        R[i + 1, j + 1] = acc_fun(D[i, j], R[i, j + 1], R[i + 1, j], R[i, j], args)

    return _cuda_diag_acc_matrix_from_dist_matrix


def __make_cuda_acc_matrix_from_dist_matrix(acc_fun):

    @cuda.jit
    def _cuda_acc_matrix_from_dist_matrix(D, R, mask, args=None):
        # block_index = cuda.blockIdx.x # -> TO COMPUTE SEVERAL IN // -> cdist

        thread_index = cuda.threadIdx.x

        # i = pos
        i = thread_index

        max_i = D.shape[0]
        max_j = D.shape[1]

        n_passes = max_i + max_j - 1

        for p in range(n_passes):

            j = max(0, p - i)

            if i + j == p and i < max_i and j < max_j and mask[i, j]:
                R[i + 1, j + 1] = acc_fun(D[i, j], R[i, j + 1], R[i + 1, j], R[i, j], args)

            cuda.syncthreads()

    return _cuda_acc_matrix_from_dist_matrix


@cuda.jit
def softdtw_acc_fun(d, r0, r1, r2, gamma):
    r0 = - r0 / gamma
    r1 = - r1 / gamma
    r2 = - r2 / gamma
    r_max = max(r0, r1, r2)
    r_sum = (
            math.exp(r0 - r_max) +
            math.exp(r1 - r_max) +
            math.exp(r2 - r_max)
    )
    softmin = -gamma * (math.log(r_sum) + r_max)
    return d + softmin


@cuda.jit
def frechet_acc_fun(d, r0, r1, r2, _):
    return max(d, min(r0, r1, r2))


@cuda.jit
def dtw_acc_fun(d, r0, r1, r2, _):
    return d + min(r0, r1, r2)


soft_dtw_cuda_acc_matrix_from_dist_matrix = __make_cuda_acc_matrix_from_dist_matrix(
    acc_fun=softdtw_acc_fun
)

dtw_cuda_acc_matrix_from_dist_matrix = __make_cuda_acc_matrix_from_dist_matrix(
    acc_fun=dtw_acc_fun
)

frechet_cuda_acc_matrix_from_dist_matrix = __make_cuda_acc_matrix_from_dist_matrix(
    acc_fun=frechet_acc_fun
)

soft_dtw_cuda_diag_acc_matrix_from_dist_matrix = __make_diag_cuda_acc_matrix_from_dist_matrix(
    acc_fun=softdtw_acc_fun
)

dtw_cuda_diag_acc_matrix_from_dist_matrix = __make_diag_cuda_acc_matrix_from_dist_matrix(
    acc_fun=dtw_acc_fun
)

frechet_cuda_diag_acc_matrix_from_dist_matrix = __make_diag_cuda_acc_matrix_from_dist_matrix(
    acc_fun=frechet_acc_fun
)


def _soft_dtw_cuda(
        s1,
        s2,
        gamma,
        global_constraint=0,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
):
    m, n = s1.shape[0], s2.shape[0]
    mask = _compute_mask(m, n, global_constraint, sakoe_chiba_radius, itakura_max_slope).to(s1.device)

    D = torch.cdist(s1, s2) ** 2

    R = D.new_full((m + 1, n + 1), torch.inf)
    R[0, 0] = 0.

    max_threads_per_block = torch.cuda.get_device_properties().max_threads_per_block

    D = cuda.as_cuda_array(D.detach().contiguous())
    R = cuda.as_cuda_array(R.detach().contiguous())
    mask = cuda.as_cuda_array(mask)

    if max(m, n) > max_threads_per_block:
        for diag_index in range(n + m - 1):
            i_min_diag = max(0, diag_index - n + 1)
            i_max_diag = min(m - 1, diag_index)
            diag_length = i_max_diag - i_min_diag + 1

            n_threads = max_threads_per_block
            n_blocks = math.ceil(diag_length / n_threads)

            soft_dtw_cuda_diag_acc_matrix_from_dist_matrix[n_blocks, n_threads](D, R, mask, diag_index, gamma)

        return R[-1, -1]

    n_threads = min(max_threads_per_block, max(m, n))
    n_blocks = math.ceil(max(m, n) / max_threads_per_block)
    soft_dtw_cuda_acc_matrix_from_dist_matrix[n_blocks, n_threads](D, R, mask, gamma)
    return R[-1, -1]


def _dtw_cuda(
        s1,
        s2,
        global_constraint=0,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
):
    m, n = s1.shape[0], s2.shape[0]
    mask = _compute_mask(m, n, global_constraint, sakoe_chiba_radius, itakura_max_slope).to(s1.device)

    D = torch.cdist(s1, s2) ** 2

    R = D.new_full((m + 1, n + 1), torch.inf)
    R[0, 0] = 0.

    max_threads_per_block = torch.cuda.get_device_properties().max_threads_per_block

    D = cuda.as_cuda_array(D.detach().contiguous())
    R = cuda.as_cuda_array(R.detach().contiguous())
    mask = cuda.as_cuda_array(mask)

    if max(m, n) > max_threads_per_block:
        for diag_index in range(n + m - 1):
            i_min_diag = max(0, diag_index - n + 1)
            i_max_diag = min(m - 1, diag_index)
            diag_length = i_max_diag - i_min_diag + 1

            n_threads = max_threads_per_block
            n_blocks = math.ceil(diag_length / n_threads)

            dtw_cuda_diag_acc_matrix_from_dist_matrix[n_blocks, n_threads](D, R, mask, diag_index, 0)

        return R[-1, -1]

    n_threads = min(max_threads_per_block, max(m, n))
    n_blocks = math.ceil(max(m, n) / max_threads_per_block)
    dtw_cuda_acc_matrix_from_dist_matrix[n_blocks, n_threads](D, R, mask, 0)
    return math.sqrt(R[-1, -1])


def _frechet_cuda(
        s1,
        s2,
        global_constraint=0,
        sakoe_chiba_radius=None,
        itakura_max_slope=None,
):
    m, n = s1.shape[0], s2.shape[0]
    mask = _compute_mask(m, n, global_constraint, sakoe_chiba_radius, itakura_max_slope).to(s1.device)

    D = torch.cdist(s1, s2) ** 2

    R = D.new_full((m + 1, n + 1), torch.inf)
    R[0, 0] = 0.

    D = cuda.as_cuda_array(D.detach().contiguous())
    R = cuda.as_cuda_array(R.detach().contiguous())
    mask = cuda.as_cuda_array(mask)

    max_threads_per_block = torch.cuda.get_device_properties().max_threads_per_block

    if max(m, n) > max_threads_per_block:
        for diag_index in range(n + m - 1):
            i_min_diag = max(0, diag_index - n + 1)
            i_max_diag = min(m - 1, diag_index)
            diag_length = i_max_diag - i_min_diag + 1

            n_threads = max_threads_per_block
            n_blocks = math.ceil(diag_length / n_threads)

            frechet_cuda_diag_acc_matrix_from_dist_matrix[n_blocks, n_threads](D, R, mask, diag_index, 0)

        return R[-1, -1]

    n_threads = min(max_threads_per_block, max(m, n))
    n_blocks = math.ceil(max(m, n) / max_threads_per_block)
    frechet_cuda_acc_matrix_from_dist_matrix[n_blocks, n_threads](D, R, mask, 0)

    return R[-1, -1]
