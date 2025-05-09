import logging

import torch
from torch.nn.functional import pad

from typing import Tuple, Callable
from enum import Enum, auto

from pymritools.recon.loraks_dev.ps_loraks import OperatorType
from pymritools.recon.loraks_dev_cleanup.loraks import LoraksBase, LoraksOptions
from pymritools.recon.loraks_dev_cleanup.matrix_indexing import get_circular_nb_indices
from pymritools.recon.loraks_dev_cleanup.operators import S, C, Operator

log_module = logging.getLogger(__name__)


class SolverType(Enum):
    LEASTSQUARES = auto()
    AUTOGRAD = auto()


def get_ac_matrix_s(k_data: torch.Tensor, s: S):
    # we extract the neighborhood size
    nb_size = s.matrix_size[0] * k_data.shape[0]
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


def create_loss_function(
        operator: Callable[[torch.Tensor, torch.Tensor, Tuple], torch.Tensor],
        rank: int, lambda_factor: float, solver: SolverType):
    match solver:
        case SolverType.LEASTSQUARES:
            def loss_func(
                    k_sampled_points: torch.Tensor,
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




    def loss_func(k_space_candidate: torch.Tensor,
                  indices: torch.Tensor,
                  matrix_shape: Tuple,
                  k_sampled_points: torch.Tensor,
                  sampling_mask: torch.Tensor,
                  lam_s: float,
                  device: torch.device = "cpu"):
        # lambda <= 0.0 means true data consistency
        if lam_s > 0.0:
            k_data = k_space_candidate
        else:
            k_data = k_sampled_points.clone()
            k_data[~sampling_mask] = k_space_candidate
        matrix = operator(k_data, indices, matrix_shape)
        u, s, vh = svd_func(matrix)
        s_r = threshold_func(s)
        matrix_recon_loraks = torch.matmul(u * s_r.to(u.dtype), vh)
        loss_1 = torch.linalg.norm(matrix - matrix_recon_loraks, ord="fro")
        if lam_s > 0.0:
            loss_2 = torch.linalg.norm(k_space_candidate * sampling_mask - k_sampled_points)
            return loss_2 + lam_s * loss_1, loss_1, loss_2
        else:
            return loss_1, loss_1, 0.0

    # return torch.compile(loss_func, fullgraph=True)
    return loss_func


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

        if self.fast_compute:
            # set fft algorithm
            self._log("Using fast computation via FFTs")
        else:
            self._log_error("Using slow compute not yet implemented")
            self._log("Using operator based (slower) algorithm")

        # in the end we need an M operator and a b vector to solve Mf = b least squares via cgd
        # self.count_matrix = self._get_count_matrix()
        torch.cuda.empty_cache()

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

    def _solve_cgd(self, k_sampled_points: torch.Tensor, vs: torch.Tensor, vc: torch.Tensor):
        # choose M and b operator based on true data consistency
        # solve cgd

    def _solve_autograd(self, k_sampled_points: torch.Tensor, vs: torch.Tensor, vc: torch.Tensor):
        # build candidate
        log_module.info("Init optimization")
        mask = k_sampled_points.abs() > 1e-10
        k_init = torch.randn_like(k_sampled_points) * 1e-5

        #

        k_data_consistency = k_sampled_points[mask]
        k_init[mask] = k_data_consistency
        k = k_init.clone().to(device).requires_grad_()
        k_data_consistency = k_data_consistency.to(device)


    def _reconstruct_batch(self, batch):
        # steps needed in AC LORAKS per batch
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

        # choose either to solve via M and b using cgd

        # or by solving k directly using autograd

        # 5) done


        raise NotImplementedError("Not yet implemented")


