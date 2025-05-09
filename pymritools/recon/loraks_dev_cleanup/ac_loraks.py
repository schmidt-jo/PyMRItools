import logging

import torch
from torch.nn.functional import pad

from enum import Enum, auto

from pymritools.recon.loraks_dev.ps_loraks import OperatorType
from pymritools.recon.loraks_dev_cleanup.loraks import LoraksBase, LoraksOptions
from pymritools.recon.loraks_dev_cleanup.matrix_indexing import get_circular_nb_indices

log_module = logging.getLogger(__name__)


def get_ac_matrix(k_data: torch.Tensor, indices: torch.Tensor, indices_rev: torch.Tensor, matrix_shape: tuple, operator: callable):
    nb_size = matrix_shape[0] * k_data.shape[0]
    mask = torch.abs(k_data) > 1e-10

    ac_matrix = operator(k_data)

    mask_p = torch.reshape(mask.view(*k_data.shape[:-2], -1)[:, indices], (-1, indices.shape[-1]))
    mask_f = torch.reshape(mask.view(*k_data.shape[:-2], -1)[:, indices_rev], (-1, indices.shape[-1]))

    idx = (torch.sum(mask_p, dim=0) == nb_size) & (torch.sum(mask_f, dim=0) == nb_size)
    idx = torch.concatenate([idx, idx], dim=0)
    ac_matrix[:, ~idx] = 0.0
    return ac_matrix


def get_count_matrix(shape_batch: tuple, indices: torch.Tensor,
                     operator: callable, operator_adjoint: callable, device: torch.device) -> torch.Tensor:
    in_ones = torch.ones(shape_batch, device=device, dtype=torch.complex64)
    ones_matrix = operator(k_space=in_ones)
    count_matrix = operator_adjoint(
        ones_matrix,
        indices=indices.to(device), k_space_dims=shape_batch
    ).real.to(torch.int)
    return count_matrix


def get_nullspace(m_ac: torch. Tensor, rank: int = 150):
    mmh = m_ac @ m_ac.mH
    e_vals, e_vecs = torch.linalg.eigh(mmh, UPLO="U")
    idx = torch.argsort(torch.abs(e_vals), descending=True)
    um = e_vecs[:, idx]
    e_vals = e_vals[idx]
    return um[:, rank:].mH, e_vals


def complex_subspace_representation(v: torch.Tensor, matrix_nb_size: int):
    nfilt, filt_size = v.shape
    nss_c = torch.reshape(v, (nfilt, -1, matrix_nb_size))
    nss_c = nss_c[:, ::2] + 1j * nss_c[:, 1::2]
    return torch.reshape(nss_c, (nfilt, -1))


def embed_circular_patch(v: torch.Tensor, nb_radius: int):
    nfilt, filt_size = v.shape

    # get indices
    circular_nb_indices = get_circular_nb_indices(nb_radius=nb_radius).to(v.device)
    # find neighborhood size
    nb_size = circular_nb_indices.shape[0]

    # build squared patch
    v = torch.reshape(v, (nfilt, -1, nb_size))
    nc = v.shape[1]

    v_patch = torch.zeros(
        (nfilt, nc, nb_radius, nb_radius),
        dtype=v.dtype, device=v.device
    )
    v_patch[:, :, circular_nb_indices[:, 0], circular_nb_indices[:, 1]] = v

    return v_patch


def zero_phase_filter(v: torch.Tensor, nb_size: int, operator_type: OperatorType = OperatorType.S):
    # Conjugate of filters
    cfilt = torch.conj(v)

    # Determine ffilt based on opt
    if operator_type == OperatorType.S:
        ffilt = torch.conj(v)
    else:  # for C matrix
        ffilt = torch.flip(v, dims=(-2, -1))

    # Perform 2D FFT
    ccfilt = torch.fft.fft2(
        cfilt,
        dim=(-2, -1),
        s=(
            2 * nb_size - 1,
            2 * nb_size - 1
        )
    )
    fffilt = torch.fft.fft2(
        ffilt,
        dim=(-2, -1),
        s=(
            2 * nb_size - 1,
            2 * nb_size - 1
        )
    )

    # Compute patch via inverse FFT of element-wise multiplication and sum
    pre_patch = torch.zeros(
        (ccfilt.shape[1], fffilt.shape[1], *ccfilt.shape[-2:]),
        dtype=ccfilt.dtype, device=ccfilt.device
    )
    for idx_f in range(fffilt.shape[0]):
        pre_patch += ccfilt[idx_f].unsqueeze(1) * fffilt[idx_f].unsqueeze(0)
    patch = torch.fft.ifft2(pre_patch, dim=(-2, -1))
    return patch


def v_pad(v_patch: torch.Tensor, batch_shape: tuple, nb_size: int):
    # assumed dims of v_patch [px, py, nce, nce]
    pad_x = batch_shape[-1] - nb_size
    pad_y = batch_shape[-2] - nb_size
    return pad(
        v_patch,
        (
            0, pad_x,
            0, pad_y
        ),
        mode='constant', value=0.0
    )

def v_shift(v_pad: torch.Tensor, batch_shape: tuple, nb_size: int, operator_type: OperatorType = OperatorType.S):
    if operator_type == OperatorType.S:
        return torch.roll(
            v_pad,
            dims=(-2, -1),
            shifts=(
                -2 * nb_size + 2 - batch_shape[-2] % 2,
                -2 * nb_size + 2 - batch_shape[-1] % 2
            )
        )
    else:
        return torch.roll(
            v_pad,
            dims=(-2, -1),
            shifts=(
                - nb_size + 1,
                - nb_size + 1
            )
        )


class SolverType(Enum):
    LEASTSQUARES = auto()
    AUTOGRAD = auto()

class AcLoraksOptions(LoraksOptions):
    def __init__(self):
        super().__init__()
        self.fast_compute: bool = True
        self.solver_type: SolverType = SolverType.LEASTSQUARES


class AcLoraks(LoraksBase):
    def __init__(self):
        super().__init__()

    def configure(self, options: AcLoraksOptions):
        self.options = options
        self.device = options.device
        self.regularization_lambda = options.regularization_lambda
        self.loraks_neighborhood_radius = options.loraks_neighborhood_radius
        self.loraks_matrix_type = options.loraks_matrix_type
        self.max_num_iter = options.max_num_iter
        self.fast_compute = options.fast_compute
        self.solver_type = options.solver_type
        self.rank = options.rank
        self.tol = 1e-5

    # steps needed in AC LORAKS
    # 1) deduce AC region within Loraks Matrix
    # 2) extract nullspace based on rank
    # 3) compute filtered nullspace kernel to reduce memory demands -> v
    # 4) set loss function to be simple matrix norm minimization of ||M v||,
    # can be computed as multiplication in Fourier space
    # 5) done
    def _reconstruct_batch(self, batch):
        raise NotImplementedError("Not yet implemented")


