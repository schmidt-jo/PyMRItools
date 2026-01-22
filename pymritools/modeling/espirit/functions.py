"""
ESPirit functions.
Increased computational performance, avoiding loops, gpu enabled (not tested).
Sensitivity map estimation based on:
Uecker et al. ESPIRiT--an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA.
Magn Reson Med. 2014 Mar;71(3):990-1001. doi: 10.1002/mrm.24751.

"""
import logging
import time

import torch
import numpy as np
import tqdm
from timeit import Timer
from torch.nn.functional import pad
from pymritools.config import setup_program_logging, setup_parser
from pymritools.utils import fft_to_img

logger = logging.getLogger(__name__)


def get_hankel_indices(nx: int, ny: int, nz: int, kernel_size: int):
    idx_spatial = torch.Tensor([
        [x, y, z] for x in range(max(1, nx - kernel_size + 1)) for y in range(max(1, ny - kernel_size + 1)) for z in range(max(1, nz - kernel_size + 1))
    ]).to(torch.int)
    idx_neighborhood = torch.Tensor([
        [x, y, z] for x in range(min(nx, kernel_size)) for y in range(min(ny, kernel_size)) for z in range(min(nz, kernel_size))
    ]).to(torch.int)
    idxs = idx_spatial[:, None, :] + idx_neighborhood[None, :, :]
    return idxs


def built_hankel(k_ac: torch.Tensor, kernel_size: int):
    hankel_indices = get_hankel_indices(
        nx=k_ac.shape[0], ny=k_ac.shape[1],
        nz=k_ac.shape[2], kernel_size=kernel_size)
    matrix = k_ac[hankel_indices[:, :, 0], hankel_indices[:, :, 1], hankel_indices[:, :, 2], :]
    return matrix.reshape(matrix.shape[0], -1)


def get_ker_imgs(kernels: torch.Tensor, kernel_size: int):
    # flip and conjugate
    kernels = kernels.flip(0, 1, 2).conj()
    ker_imgs = fft_to_img(kernels, dims=(0, 1, 2), norm="ortho") * np.sqrt(
                np.prod(kernels.shape[:3])
            ) / np.sqrt(kernel_size ** np.sum(np.array(kernels.shape[:3]) > 1))
    return ker_imgs


def map_estimation(
        k_rpsc: torch.Tensor, kernel_size: int, num_ac_lines: int,
        rank_fraction_ac_matrix: float = 0.01, eigenvalue_cutoff: float = 0.99,
        device: torch.device = torch.get_default_device()
):
    """
    Derives the ESPIRiT operator (maps).

    Arguments:
      k_rpsc: Multi channel k-space data. Expected dimensions are (n_read, n_phase, n_slice, n_channels)
         if there's an additional dimension present, It's assumed to be echoes and averaged over,
         but crucially assumed to be the last dimension!
      kernel_size: Parameter that determines the k-space kernel size. If k has dimensions (1, 256, 256, 8, 1), then the kernel
         will have dimensions (1, k, k, 8, 1), the kernel will always distribute only over the
      num_ac_lines: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the
         calibration region will have dimensions (1, r, r, 8)
      rank_fraction_ac_matrix: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      eigenvalue_cutoff: Crop threshold that determines eigenvalues "=1".
      device: torch device on which to perform computations.
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (nc, n_read, n_phase, n_slice, n_channels) with (idx, ...)
            being the idx'th set of ESPIRiT maps.
    """

    if k_rpsc.ndim < 4 or k_rpsc.ndim > 5:
        err = f"input tensor must have 4 dimensions (a fifth would be averaged), but got {k_rpsc.ndim}"
        logger.error(err)
        raise AttributeError(err)
    elif k_rpsc.ndim > 4:
        msg = f"input tensor has 5 dimensions, averaging over the last one assuming temporal dimension."
        logger.warning(msg)
        k_rpsc = torch.mean(k_rpsc, dim=-1)

    if kernel_size % 2 == 1:
        msg = f"kernel size must be even, got {kernel_size}, updating to next higher"
        logger.warning(msg)
        kernel_size += 1

    n_read, n_phase, n_slice, n_channels = k_rpsc.shape

    ac_x = (n_read // 2 - num_ac_lines // 2, n_read // 2 + num_ac_lines // 2) if (n_read > 1) else (0, 1)
    ac_y = (n_phase // 2 - num_ac_lines // 2, n_phase // 2 + num_ac_lines // 2) if (n_phase > 1) else (0, 1)
    ac_z = (n_slice // 2 - num_ac_lines // 2, n_slice // 2 + num_ac_lines // 2) if (n_slice > 1) else (0, 1)

    logger.info(f"Extract calibration region.")
    k_ac = k_rpsc[ac_x[0]:ac_x[1], ac_y[0]:ac_y[1], ac_z[0]:ac_z[1], :].to(torch.complex64)
    logger.info("built hankel matrix")
    matrix_a = built_hankel(k_ac=k_ac, kernel_size=kernel_size)

    logger.info("Take the Singular Value Decomposition.")
    u, s, vh = torch.linalg.svd(matrix_a, full_matrices=False)
    v = vh.mH

    logger.info("Select kernels and process.")
    n = torch.sum(s >= rank_fraction_ac_matrix * s[..., 0])
    v = v[:, :n]

    pad_x = (n_read // 2 - kernel_size // 2 + n_read % 2, n_read // 2 - kernel_size // 2) if (n_read > 1) else (0, 0)
    pad_y = (n_phase // 2 - kernel_size // 2 + n_phase % 2, n_phase // 2 - kernel_size // 2) if (n_phase > 1) else (0, 0)
    pad_z = (n_slice // 2 - kernel_size // 2 + n_slice % 2, n_slice // 2 - kernel_size // 2) if (n_slice > 1) else (0, 0)

    # Reshape into k-space kernel, flips it and takes the conjugate
    ker_dims = (
        kernel_size if n_read > 1 else 1, kernel_size if n_phase > 1 else 1, kernel_size if n_slice > 1 else 1,
        n_channels, n
    )
    kernels = torch.reshape(v, ker_dims)
    kernels = pad(kernels, (0, 0, 0, 0, *pad_z, *pad_y, *pad_x), mode='constant', value=0.0)

    del u, s, vh, v
    torch.cuda.empty_cache()

    ker_imgs = get_ker_imgs(kernels, kernel_size)

    del kernels
    torch.cuda.empty_cache()

    logger.info(f"Compute point-wise eigenvalue decomposition and maps")
    u, s, _ = torch.linalg.svd(ker_imgs, full_matrices=True)
    mask = s**2 > eigenvalue_cutoff
    u[~mask.unsqueeze(-2).expand_as(u)] = 0.0

    return u.movedim(-1, 0)


if __name__ == '__main__':
    setup_program_logging("ESPirit sensitivity map estimation", logging.INFO)