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

    logger.info("Select kernels and process.")
    n = torch.sum(s >= rank_fraction_ac_matrix * s[..., 0])
    # v = v[..., :n]
    #
    # pad_x = (n_read // 2 - kernel_size // 2 + n_read % 2, n_read // 2 - kernel_size // 2) if (n_read > 1) else (0, 0)
    # pad_y = (n_phase // 2 - kernel_size // 2 + n_phase % 2, n_phase // 2 - kernel_size // 2) if (n_phase > 1) else (0, 0)
    # pad_z = (n_slice // 2 - kernel_size // 2 + n_slice % 2, n_slice // 2 - kernel_size // 2) if (n_slice > 1) else (0, 0)
    #
    # # Reshape into k-space kernel, flips it and takes the conjugate
    # ker_dims = (
    #     kernel_size if n_read > 1 else 1, kernel_size if n_phase > 1 else 1, kernel_size if n_slice > 1 else 1,
    #     n_channels, n
    # )
    # kernels = torch.reshape(v, ker_dims)
    # kernels = pad(kernels, (0, 0, 0, 0, *pad_z, *pad_y, *pad_x), mode='constant', value=0.0)

    kx = kernel_size if n_read > 1 else 1
    ky = kernel_size if n_phase > 1 else 1
    kz = kernel_size if n_slice > 1 else 1

    kernels_small = vh[:n].reshape(n, kx, ky, kz, n_channels).permute(1, 2, 3, 4, 0)
    kernels_small = kernels_small.flip(0, 1, 2).conj()

    x0 = n_read // 2 - kx // 2 + n_read % 2 if n_read > 1 else 0
    y0 = n_phase // 2 - ky // 2 + n_phase % 2 if n_phase > 1 else 0
    z0 = n_slice // 2 - kz // 2 + n_slice % 2 if n_slice > 1 else 0

    kernels = torch.zeros(
        (n_read, n_phase, n_slice, n_channels, n),
        dtype=kernels_small.dtype,
        device=kernels_small.device,
    )
    kernels[x0:x0 + kx, y0:y0 + ky, z0:z0 + kz, :, :] = kernels_small

    del kernels_small, u, s, vh

    scale = (
            np.sqrt(n_read * n_phase * n_slice) /
            np.sqrt(kernel_size ** sum(dim > 1 for dim in (n_read, n_phase, n_slice)))
    )

    ker_imgs = []
    for idx_l in tqdm.trange(n_channels, desc="Batch processing FFT"):
        ker_imgs.append((fft_to_img(kernels[..., idx_l, :].to(device), dims=(0, 1, 2), norm="ortho") * scale).cpu())
    ker_imgs = torch.stack(ker_imgs, dim=-2)
    del kernels, scale
    torch.cuda.empty_cache()

    # kernels = kernels.flip(0, 1, 2).conj()
    # ker_imgs = fft_to_img(kernels, dims=(0, 1, 2), norm="ortho") * np.sqrt(
    #             np.prod(kernels.shape[:3])
    #         ) / np.sqrt(kernel_size ** np.sum(np.array(kernels.shape[:3]) > 1))

    # del kernels

    logger.info(f"Compute point-wise eigenvalue decomposition and maps")
    bar = tqdm.trange(ker_imgs.shape[0], desc="Batch processing pointwise SVD")
    b = ker_imgs.shape[1]
    collect_u = []
    for idx_x in bar:
        uu = []
        for idx_y in range(b):
            batch = ker_imgs[idx_x, idx_y, :, :, :]
            batch = batch / torch.linalg.norm(batch)
            u, s, _ = torch.linalg.svd(batch, full_matrices=True)
            mask = s ** 2 > eigenvalue_cutoff
            u[~mask.unsqueeze(-2).expand_as(u)] = 0.0

            name = "%" * (idx_y + 1)
            bar.set_postfix({"step": f"{name.ljust(b, '_')}"})

            uu.append(u)
        collect_u.append(torch.stack(uu, dim=0))
    u = torch.stack(collect_u, dim=0)
    return u.movedim(-1, 0)
