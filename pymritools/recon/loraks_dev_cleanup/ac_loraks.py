import logging
import tqdm
from enum import Enum, auto

import torch
from torch.nn.functional import pad
from dataclasses import dataclass
from typing import Tuple, Callable

from pymritools.recon.loraks_dev_cleanup.loraks import LoraksBase, LoraksOptions, OperatorType, LoraksImplementation
from pymritools.recon.loraks_dev_cleanup.matrix_indexing import get_circular_nb_indices
from pymritools.recon.loraks_dev_cleanup.operators import S, C
from pymritools.recon.loraks_dev_cleanup.utils import prepare_k_space_to_batches, unprepare_batches_to_k_space
from pymritools.utils.algorithms import cgd

log_module = logging.getLogger(__name__)


class SolverType(Enum):
    LEASTSQUARES = auto()
    AUTOGRAD = auto()


class ComputationType(Enum):
    FFT = auto()
    REGULAR = auto()


def get_ac_matrix_s(k_data: torch.Tensor, s: S):
    # we extract the neighborhood size
    nb_size = s.matrix_size()[0] * k_data.shape[0]
    mask = torch.abs(k_data) > 1e-10

    ac_matrix = s.operator(k_data)

    mask_p = torch.reshape(mask.view(*k_data.shape[:-2], -1)[:, s.indices], (-1, s.indices.shape[-1]))
    mask_f = torch.reshape(mask.view(*k_data.shape[:-2], -1)[:, s.indices_rev], (-1, s.indices.shape[-1]))

    idx = (torch.sum(mask_p, dim=0) == nb_size) & (torch.sum(mask_f, dim=0) == nb_size)
    idx = torch.concatenate([idx, idx], dim=0)
    ac_matrix[:, ~idx] = 0.0
    return ac_matrix


def get_ac_matrix_c(k_data: torch.Tensor, c: C):
    # we extract the neighborhood size
    nb_size = c.matrix_size[0] * k_data.shape[0]
    mask = torch.abs(k_data) > 1e-10

    ac_matrix = c.operator(k_data)

    mask_p = c.operator(mask)

    idx = (torch.sum(mask_p, dim=0) == nb_size)
    ac_matrix[:, ~idx] = 0.0
    return ac_matrix


def get_count_matrix(shape_batch: Tuple, indices: torch.Tensor,
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


def calc_v_pad(v_patch: torch.Tensor, batch_shape: Tuple, nb_size: int):
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

def calc_v_shift(v_pad: torch.Tensor, batch_shape: Tuple, nb_size: int, operator_type: OperatorType = OperatorType.S):
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

def m_op_base(x: torch.Tensor, v_c: torch.Tensor, v_s: torch.Tensor, nb_size: int, shape_batch: Tuple):
    # dims [nx, ny, nce]
    pad_x = pad(
        x,
        (0, nb_size - 1, 0, nb_size - 1),
        mode="constant", value=0.0
    )
    fft_x = torch.fft.fft2(pad_x, dim=(-2, -1))
    # dims [nx + nb - 1, ny + nb - 1, nce]
    mv_c = torch.sum(v_c * fft_x.unsqueeze(0), dim=1)
    mv_s = torch.sum(v_s * torch.conj(fft_x).unsqueeze(0), dim=1)

    imv_c = torch.fft.ifft2(mv_c, dim=(-2, -1))[..., :shape_batch[-2], :shape_batch[-1]]
    imv_s = torch.fft.ifft2(mv_s, dim=(-2, -1))[..., :shape_batch[-2], :shape_batch[-1]]

    return imv_c - imv_s

def get_m_operator_fft(
        v_s: torch.Tensor, v_c: torch.Tensor, mask: torch.Tensor, nb_size: int,
        shape_batch: Tuple, lambda_factor: float, device: torch.device):
    # fast computation using ffts
    if lambda_factor > 0.0:
        aha = torch.zeros(shape_batch, dtype=v_s.dtype, device=device)
        aha[mask] = 1.0

        # x has shape [mx, ny, nc, ne]

        def _m_op(x):
            x = torch.reshape(x, shape_batch)
            m = m_op_base(x, v_c=v_c, v_s=v_s, nb_size=nb_size, shape_batch=shape_batch)
            return (aha * x + 2 * lambda_factor * m)

    else:
        def _m_op(x):
            tmp = torch.zeros(shape_batch, dtype=v_s.dtype, device=device)
            tmp[~mask] = x
            m = m_op_base(tmp, v_c=v_c, v_s=v_s)
            return 2 * m[~mask]

    return _m_op


def get_m_operator(
        vvh: torch.Tensor, mask: torch.Tensor, count_matrix: torch.Tensor,
        operator: callable, operator_adjoint: callable, shape_batch: Tuple,
        lambda_factor: float, device: torch.device):
    # original algorithm
    if lambda_factor > 0.0:
        aha = (
            # self.sampling_mask_xyt.unsqueeze(-2).to(self.device) +
                mask.to(dtype=count_matrix.dtype, device=device) +
                lambda_factor * count_matrix
        )

        def _m_op(x):
            m_v = operator_adjoint(operator(x) @ vvh)
            return aha * x - lambda_factor * m_v
    else:

        def _m_op(x):
            m = count_matrix[~mask] * x

            tmp = torch.zeros(shape_batch, dtype=vvh.dtype, device=vvh.device)
            tmp[~mask] = x
            mvs = operator_adjoint(operator(tmp) @ vvh)[~mask]
            return m - mvs

    return _m_op

def get_b_vector_fft(k: torch.Tensor, mask: torch.Tensor, v_s: torch.Tensor, v_c: torch.Tensor,
                     lambda_factor: float, nb_size: int, shape_batch: Tuple):
    if lambda_factor > 0.0:
        return k
    else:
        m = m_op_base(k, v_c=v_c, v_s=v_s, nb_size=nb_size, shape_batch=shape_batch)
        return - 2 * m[~mask]


def get_b_vector(k: torch.Tensor, mask:torch.Tensor, vvh: torch.Tensor,
                 operator: callable, operator_adjoint: callable,
                 lambda_factor: float):
    if lambda_factor > 0.0:
        return k
    else:
        return operator_adjoint(operator(k) @ vvh)[~mask]

def embedd_data(k_sampled_points: torch.Tensor, k_candidate: torch.Tensor,
                mask_sampled_points: torch.Tensor, lambda_factor: float):
    if lambda_factor > 0.0:
        return k_candidate
    else:
        tmp = k_sampled_points.clone()
        tmp[~mask_sampled_points] = k_candidate
        return tmp


def solve_cgd_batch(
        k_sampled_points: torch.Tensor,
        lambda_factor: float,
        mask: torch.Tensor,
        shape_batch: Tuple,
        vs: torch.Tensor,
        vc: torch.Tensor,
        nb_size: int,
        device: torch.device = "cpu"):
        m = get_m_operator_fft(
            v_s=vs, v_c=vc, mask=mask, nb_size=nb_size,
            lambda_factor=lambda_factor, shape_batch=shape_batch
        )
        b = get_b_vector_fft(
            k=k_sampled_points, mask=mask, v_s=vs, v_c=vc,
            lambda_factor=lambda_factor, nb_size=nb_size, shape_batch=shape_batch
        )
        k_opt = cgd(func_operator=m, x=torch.zeros_like(b), b=b)
        return k_opt.cpu()


def solve_autograd_batch(
        k_sampled_points: torch.Tensor,
        lambda_factor: float,
        mask: torch.Tensor,
        shape_batch: Tuple,
        vs: torch.Tensor,
        vc: torch.Tensor,
        nb_size: int,
        device: torch.device = "cpu",
        max_num_iter: int = 200,
        learning_rate_function: Callable = lambda x: 1e-3):
        log_module.info("Init autograd solver")
        k_init = torch.randn_like(k_sampled_points) * 1e-5
        k_data_consistency = k_sampled_points[mask]
        k_init[mask] = k_data_consistency
        k = k_init.clone().to(device).requires_grad_()
        k_data_consistency = k_data_consistency.to(device)

        progress_bar = tqdm.trange(max_num_iter, desc="Optimization")

        # iterations
        for i in progress_bar:
            # embed data based on lambda factor
            k_data = embedd_data(
                k_sampled_points=k_sampled_points, mask_sampled_points=mask,
                k_candidate=k, lambda_factor=lambda_factor
            )
            # solve || mv || in fourier space
            mv = m_op_base(
                x=k_data, v_c=vc, v_s=vs, nb_size=nb_size, shape_batch=shape_batch
            )
            loss_lr = torch.linalg.norm(mv, ord="fro")
            if lambda_factor > 0.0:
                loss_dc = torch.linalg.norm(k_data - k_data_consistency)
                loss = loss_dc + lambda_factor * loss_lr
            else:
                loss_dc = 0.0
                loss = loss_lr

            loss.backward()
            # Use the optimal learning_rate to update parameters
            with torch.no_grad():
                k -= learning_rate_function(i) * k.grad
            k.grad.zero_()

            progress_bar.postfix = (
                f"LR: {learning_rate_function(i):.6f} -- "
                f"LowRank Loss: {1e3 * loss_lr.item():.2f} -- "
                f"Data Loss: {1e3 * loss_dc.item():.2f} -- "
                f"Total Loss: {1e3 * loss.item():.2f}"
            )

        # k is our converged best guess candidate, need to unwrap / reshape
        return k.detach().cpu()


@dataclass
class AcLoraksOptions(LoraksOptions):
    loraks_type: LoraksImplementation = LoraksImplementation.AC_LORAKS
    computation_type: ComputationType = ComputationType.FFT
    solver_type: SolverType = SolverType.LEASTSQUARES
    loraks_neighborhood_radius: int = 3


class AcLoraks(LoraksBase):
    def __init__(self):
        super().__init__()
        self.op = None
        self.tol = None
        self.rank = None
        self.solver_type = None
        self.computation_type = None
        self.max_num_iter = None
        self.loraks_matrix_type = None
        self.loraks_neighborhood_radius = None
        self.regularization_lambda = None
        self.device = None
        self.options = None

    def configure(self, options: AcLoraksOptions = AcLoraksOptions()):
        self.options = options
        self.device = options.device
        self.regularization_lambda = options.regularization_lambda
        self.loraks_neighborhood_radius = options.loraks_neighborhood_radius
        self.loraks_matrix_type = options.loraks_matrix_type
        self.max_num_iter = options.max_num_iter
        self.computation_type = options.computation_type
        self.solver_type = options.solver_type
        self.rank = options.rank.value
        self.tol = 1e-5

        if self.loraks_matrix_type == OperatorType.S:
            self.op = S
        else:
            self.op = C

        if self.regularization_lambda < 1e-9:
            # set strict data consistency
            self._log("Using strict data consistency algorithm")
        else:
            # set regularization version
            self._log(f"Using regularized algorithm (lambda: {self.regularization_lambda:.3f})")

        if self.computation_type == ComputationType.FFT:
            # set fft algorithm
            self._log("Using fast computation via FFTs")
        else:
            self._log_error("Using slow compute not yet implemented")
            self._log("Using operator based (slower) algorithm")

        # in the end we need an M operator and a b vector to solve Mf = b least squares via cgd
        # self.count_matrix = self._get_count_matrix()
        torch.cuda.empty_cache()

    def _prep_k_space_to_batches(self, k_space: torch.Tensor):
        """

        The input tensor must have the shape [nt, ne, nz, ny, nx].
        The output tensor will be reshaped into
        batches with a fixed output dimensionality of [b, -1], where b is the batch size.

        :param k_space: Input k-space tensor of shape [nt, ne, nz, ny, nx].
        :raises ValueError: If the input tensor does not match the expected shape.
        :return: Output tensor divided into batches, reshaped to [b, -1].
        """

    def _prepare_batch(self, batch):
        # 1) deduce AC region within Loraks Matrix
        m_ac = self._get_ac_matrix(
            batch=batch,
        )
        # 2) extract nullspace based on rank
        v, _ = get_nullspace(m_ac, rank=self.rank)
        # 3) compute filtered nullspace kernel to reduce memory demands -> v
        # complexify nullspace
        v = self._complex_subspace_representation(v)
        torch.cuda.empty_cache()

        # prep zero phase filter input
        v_patch = self._embed_circular_patch(v)
        del v
        torch.cuda.empty_cache()

        # fft
        vs = self._get_v_fft(v_cplx_embed=v_patch, batch_shape=batch.shape, operator_type=OperatorType.S)
        vc = self._get_v_fft(v_cplx_embed=v_patch, batch_shape=batch.shape, operator_type=OperatorType.C)

        return vc, vs

    def _get_ac_matrix(self, batch: torch.Tensor):
        if self.loraks_matrix_type == OperatorType.S:
            return get_ac_matrix_s(
                k_data=batch, s=self.op
            )
        else:
            return get_ac_matrix_c(
                k_data=batch, c=self.op
            )

    def _complex_subspace_representation(self, v):
        return complex_subspace_representation(v, matrix_nb_size=self.loraks_neighborhood_radius)

    def _embed_circular_patch(self, v):
        return embed_circular_patch(v, nb_radius=self.loraks_neighborhood_radius)

    def _get_v_fft(self, v_cplx_embed: torch.Tensor, batch_shape: Tuple, operator_type: OperatorType):
        # zero phase filter
        v_filt = zero_phase_filter(
            v_cplx_embed.clone(), nb_size=self.loraks_neighborhood_radius, operator_type=operator_type
        )
        del v_cplx_embed
        torch.cuda.empty_cache()

        # pad and fft shift
        v_shift = calc_v_shift(
            calc_v_pad(v_filt, batch_shape=batch_shape, nb_size=self.loraks_neighborhood_radius),
            batch_shape=batch_shape, nb_size=self.loraks_neighborhood_radius, operator_type=operator_type
        )
        del v_filt
        torch.cuda.empty_cache()

        # fft
        return torch.fft.fft2(v_shift, dim=(-2, -1))

    def _zero_phase_filter(self, v):
        return zero_phase_filter(v, self.loraks_neighborhood_radius, self.loraks_matrix_type)

    @staticmethod
    def _log(msg):
        msg = f"\t\t- {msg}"
        log_module.info(msg)

    @staticmethod
    def _log_error(msg):
        log_module.error(msg)
        raise AttributeError(msg)

    # we want to pick the solver
    def _solve_cgd(self, k_sampled_points: torch.Tensor, vs: torch.Tensor, vc: torch.Tensor):
        # choose M and b operator based on true data consistency
        # solve cgd
        return solve_cgd_batch(
            k_sampled_points=k_sampled_points, lambda_factor=self.regularization_lambda,
            mask=torch.abs(k_sampled_points) > 1e-10, shape_batch=k_sampled_points.shape,
            vs=vs, vc=vc, nb_size=self.loraks_neighborhood_radius, device=self.device
        )

    def _solve_autograd(self, k_sampled_points: torch.Tensor, vs: torch.Tensor, vc: torch.Tensor):
        return solve_autograd_batch(
            k_sampled_points=k_sampled_points, lambda_factor=self.regularization_lambda,
            mask=torch.abs(k_sampled_points) > 1e-10, shape_batch=k_sampled_points.shape,
            vs=vs, vc=vc, nb_size=self.loraks_neighborhood_radius, device=self.device,
            max_num_iter=self.max_num_iter, learning_rate_function=lambda x: 1e-3
        )

    def reconstruct_batch(self, batch):
        # steps needed in AC LORAKS per batch
        vc, vs = self._prepare_batch(batch=batch)

        if self.solver_type == SolverType.LEASTSQUARES:
            # choose either to solve via M and b using cgd
            k = self._solve_cgd(k_sampled_points=batch, vs=vs, vc=vc)
        else:
            # or by solving k directly using autograd
            k = self._solve_autograd(k_sampled_points=batch, vs=vs, vc=vc)

        # if true data consisctency is true embed the data
        k = embedd_data(
            k_sampled_points=batch, mask_sampled_points=torch.abs(batch) > 1e-10,
            lambda_factor=self.regularization_lambda, k_candidate=k
        )
        return k


