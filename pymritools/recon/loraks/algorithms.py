import logging
import pathlib as plib

import tqdm
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.recon.loraks.operators import (
    c_operator, c_adjoint_operator,
    s_operator, s_adjoint_operator
)
from pymritools.utils import get_idx_2d_circular_neighborhood_patches_in_shape
from pymritools.utils.algorithms import cgd

log_module = logging.getLogger(__name__)


def deduce_read_direction(sampling_mask_x_y_t):
    # deduce read direction, assuming fully sampled read
    read_dir = -1
    if torch.sum(
            torch.abs(sampling_mask_x_y_t[:, int(sampling_mask_x_y_t.shape[1] / 2), 0].to(torch.int)),
            dim=0
    ) < sampling_mask_x_y_t.shape[0]:
        read_dir = 1
    if torch.sum(
            torch.abs(sampling_mask_x_y_t[int(sampling_mask_x_y_t.shape[0] / 2), :, 0].to(torch.int)),
            dim=0
    ) < sampling_mask_x_y_t.shape[1]:
        if read_dir == 1:
            msg = f"found k - space to be undersampled in x and y direction. Can choose either direction for processing."
            log_module.info(msg)
        else:
            read_dir = 0
    return read_dir


def get_loraks_matrix_from_ones(shape: tuple, indices: torch.Tensor, mode: str):
    in_ones = torch.ones(shape, device=torch.device("cpu"), dtype=torch.complex128)
    if mode == "s":
        op = s_operator
    elif mode == "c":
        op = c_operator
    else:
        err = f"Mode {mode} not implemented."
        log_module.error(err)
        raise ValueError(err)
    return op(
            k_space_x_y_ch_t=in_ones,
            indices=indices
        )


def get_count_matrix(shape: tuple, indices: torch.Tensor, mode: str):
    if mode == "s":
        op_adj = s_adjoint_operator
    elif mode == "c":
        op_adj = c_adjoint_operator
    else:
        err = f"Mode {mode} not implemented."
        log_module.error(err)
        raise ValueError(err)
    ones_matrix = get_loraks_matrix_from_ones(shape, indices, mode)
    count_matrix = op_adj(
        s_matrix=ones_matrix,
        indices=indices, k_space_dims=shape
    ).real.to(torch.int)
    return count_matrix


def get_v_matrix_of_ac_subspace(
        k_space_x_y_ch_t: torch.Tensor, indices: torch.Tensor, ac_indices: torch.Tensor, mode: str,
        rank: int, use_eigh: bool = True
):
    if mode == "s":
        op = s_operator
    elif mode == "c":
        op = c_operator
    else:
        err = f"Mode {mode} not implemented."
        log_module.error(err)
        raise ValueError(err)
    # find the V matrix for the ac subspaces
    m_ac = op(k_space_x_y_ch_t=k_space_x_y_ch_t, indices=indices)[ac_indices]
    if use_eigh:
        # via eigh
        eig_vals, eig_vecs = torch.linalg.eigh(torch.matmul(m_ac.T, m_ac))
        m_ac_rank = eig_vals.shape[-1]
        # get subspaces from svd of subspace matrix
        eig_vals, idxs = torch.sort(torch.abs(eig_vals), descending=True)
        # eig_vecs_r = eig_vecs[idxs]
        eig_vecs = eig_vecs[:, idxs]
        # v_sub_r = eig_vecs_r[:self.rank].to(self.device)
        v_sub = eig_vecs[:, :rank]
    else:
        _, s, v = torch.linalg.svd(m_ac, full_matrices=False)
        m_ac_rank = s.shape[-1]
        v_sub = v[:, :rank]
    if m_ac_rank < rank:
        err = f"loraks rank parameter is too large, cant be bigger than ac matrix dimensions."
        log_module.error(err)
        raise ValueError(err)
    return torch.matmul(v_sub, v_sub.conj().T)


def loraks(
        k_space_x_y_z_ch_t: torch.Tensor,
        sampling_mask_x_y_t: torch.Tensor,
        radius: int,
        rank: int, lam: float,
        batch_size_echoes: int = 4,
        max_num_iter: int = 10, conv_tol: float = 1e-3,
        data_consistency = 0.95,
        device: torch.device = torch.get_default_device()):
    # __ One Time Calculations __
    # get dimensions
    shape = k_space_x_y_z_ch_t.shape
    n_read, n_phase, n_slice, n_channels, n_echoes = shape
    # for now just pick middle slice
    idx_slice = int(n_slice / 2)
    k_space_x_y_ch_t = k_space_x_y_z_ch_t[:, :, idx_slice]

    # get indices for operators
    indices = get_idx_2d_circular_neighborhood_patches_in_shape(
        shape_2d=(n_read, n_phase), nb_radius=radius, device=torch.device("cpu")
    )

    # only use S matrix for now - only calculate for relevant dims
    count_matrix = get_count_matrix(shape=(n_read, n_phase, 1, n_echoes), indices=indices, mode="s")
    mask = count_matrix > 1e-7

    # calculate matrix dimensions
    m = (n_read - radius) * (n_phase - radius)
    n = (batch_size_echoes * n_channels * indices.shape[1] * 2)

    matrix_rank = min(m, n)
    # build s_threshold based on rank
    s_threshold = torch.ones(matrix_rank, dtype=torch.float32)
    s_threshold[rank:] = 0

    # allocate result
    k = k_space_x_y_ch_t.clone().requires_grad_(True)

    # log losses
    losses = []
    bar = tqdm.trange(max_num_iter, desc="Optimization")
    for _ in bar:
        # get operator matrix
        matrix = s_operator(
            k_space_x_y_ch_t=k, indices=indices
        )

        # do svd
        # we can use torch svd, or try the randomized version, see above
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        # u, s, vh = randomized_svd(matrix, sampling_size=svd_sampling_size, oversampling_factor=5, power_projections=2)

        # threshold singular values
        s_r = s * s_threshold

        # reconstruct the low rank approximation
        matrix_recon_loraks = torch.matmul(
            torch.matmul(u, torch.diag(s_r).to(u.dtype)),
            vh
        )
        # first part of loss
        # calculate difference to low rank approx
        loss_1 = torch.linalg.norm(matrix - matrix_recon_loraks)

        # second part, calculate reconstructed k
        k_recon_loraks = s_adjoint_operator(
            s_matrix=matrix_recon_loraks, indices=indices, k_space_dims=shape
        )
        k_recon_loraks[mask] /= count_matrix[mask]

        # take difference to sampled k for samples
        loss_2 = torch.linalg.norm(k_recon_loraks * sampling_mask_x_y_t[:, :, None] - k_space_x_y_ch_t)

        loss = data_consistency * loss_2 + (1 - data_consistency) * loss_1

        loss.backward()

        with torch.no_grad():
            k -= k.grad * 0.4

        k.grad.zero_()
        # optim.step()
        # optim.zero_grad()
        losses.append(loss.item())

        bar.postfix = (
            f"loss 1: {loss_1.item():.2f} -- loss 2: {loss_2.item():.2f} -- total_loss: {loss.item():.2f} -- rank: {rank}"
        )

    return k[:, :, None]


def ac_loraks(
        k_space_x_y_z_ch_t: torch.Tensor,
        sampling_mask_x_y_t: torch.Tensor,
        radius: int,
        rank_c: int, lambda_c: float,
        rank_s: int, lambda_s: float,
        batch_size_echoes: int = 4,
        max_num_iter: int = 10, conv_tol: float = 1e-3,
        visualize: bool = True,
        device: torch.device = torch.get_default_device()):
    # __ One Time Calculations __
    read_dir = deduce_read_direction(sampling_mask_x_y_t=sampling_mask_x_y_t)

    # move read dir to front
    k_space_x_y_z_ch_t = torch.movedim(k_space_x_y_z_ch_t, read_dir, 0)
    sampling_mask_x_y_t = torch.movedim(sampling_mask_x_y_t, read_dir, 0)

    # get dimensions
    shape = k_space_x_y_z_ch_t.shape
    n_read, n_phase, n_slice, n_channels, n_echoes = shape
    # for now just pick middle slice
    idx_slice = int(n_slice / 2)
    k_space_x_y_ch_t = k_space_x_y_z_ch_t[:, :, idx_slice]

    # get indices for operators
    indices = get_idx_2d_circular_neighborhood_patches_in_shape(
        shape_2d=(n_read, n_phase), nb_radius=radius, device=torch.device("cpu")
    )

    # check if C matrix is used and calculate + extract ac indices
    if lambda_c > 1e-7:
        c_count_matrix = get_count_matrix(
            mode="c", indices=indices, shape=(n_read, n_phase, 1, n_echoes)
        )
        c_ac = sampling_mask_x_y_t[indices[..., 0], indices[..., 1]].to(torch.int)
        c_ac_idxs = torch.sum(c_ac, dim=(1, 2)) == c_ac.shape[1] * c_ac.shape[2]
    else:
        c_count_matrix = 0
        c_ac_idxs = None

    # check if S matrix is used and calculate + extract ac indices
    if lambda_s > 1e-7:
        s_count_matrix = get_count_matrix(
            mode="s", indices=indices, shape=(n_read, n_phase, 1, n_echoes)
        )
        c_ac = sampling_mask_x_y_t[indices[..., 0], indices[..., 1]].to(torch.int)
        s_ac_m = torch.flip(sampling_mask_x_y_t, dims=(0, 1))[indices[..., 0], indices[..., 1]].to(torch.int)
        s_ac_idxs = (torch.sum(c_ac, dim=(1, 2)) + torch.sum(s_ac_m, dim=(1, 2))) == c_ac.shape[1] * c_ac.shape[2] * 2
        s_ac_idxs = torch.tile(s_ac_idxs, dims=(2,))
    else:
        s_count_matrix = 0
        s_ac_idxs = None

    aha = sampling_mask_x_y_t[:, :, None].to(torch.int) + lambda_c * c_count_matrix + lambda_s * s_count_matrix

    if visualize:
        # plot count matrices
        fig = psub.make_subplots(rows=2, cols=2, subplot_titles=('C', 'S'))
        for idx_i, i in enumerate([c_count_matrix, s_count_matrix]):
            fig.add_trace(
                go.Heatmap(z=torch.abs(i)[:, :, 0, 0], showscale=False),
                row=1, col=1 + idx_i
            )
        # want to translate the indices into an image
        c_ones_matrix = get_loraks_matrix_from_ones(shape=(n_read, n_phase, 1, n_echoes), indices=indices, mode="c")
        c_ones_matrix[~c_ac_idxs] = 0
        c_ac_img = c_adjoint_operator(c_ones_matrix, indices=indices, k_space_dims=(n_read, n_phase, 1, n_echoes))
        s_ones_matrix = get_loraks_matrix_from_ones(shape=(n_read, n_phase, 1, n_echoes), indices=indices, mode="s")
        s_ones_matrix[~s_ac_idxs] = 0
        s_ac_img = s_adjoint_operator(s_ones_matrix, indices=indices, k_space_dims=(n_read, n_phase, 1, n_echoes))
        for idx_i, i in enumerate([c_ac_img, s_ac_img]):
            fig.add_trace(
                go.Heatmap(z=torch.abs(i)[:, :, 0, 0], showscale=False),
                row=2, col=1 + idx_i
            )
        fig_name = plib.Path("./examples/recon/loraks_arxv").joinpath('count-matrices_ac-region').with_suffix('.html')
        log_module.info(f"write file: {fig_name}")
        fig.write_html(fig_name.as_posix())

    # __ need to batch. we can just permute and batch echoes
    idxs_echoes = torch.randperm(n_echoes)
    num_batches = int(np.ceil(n_echoes / batch_size_echoes))

    iter_bar = tqdm.trange(num_batches, desc="batch_processing")
    for idx_b in iter_bar:
        start = idx_b * batch_size_echoes
        end = np.min([(idx_b + 1) * batch_size_echoes, n_echoes])
        batch_k_space_x_y_ch_t = k_space_x_y_ch_t[:, :, :, idxs_echoes[start:end]].to(device)
        batch_aha = aha[:, :, :, idxs_echoes[start:end]].to(device)

        # __ per slice calculations
        # ToDo: we know the ACS indices already, can we calculate this not with the full batch and masking afterwards,
        # but the other way around. i.e. masking the k_space batch with the found AC region indices and
        # then building c and s matrix?
        # __ for C
        if lambda_c > 1e-9:
            vvc = get_v_matrix_of_ac_subspace(
                k_space_x_y_ch_t=batch_k_space_x_y_ch_t, indices=indices, ac_indices=c_ac_idxs, mode="c", rank=rank_c
            )
        else:
            vvc = torch.zeros(1, device=device, dtype=torch.complex128)

        # __ for S
        if lambda_s > 1e-9:
            vvs = get_v_matrix_of_ac_subspace(
                k_space_x_y_ch_t=batch_k_space_x_y_ch_t, indices=indices, ac_indices=s_ac_idxs, mode="s", rank=rank_s
            )
        else:
            vvs = torch.zeros(1, device=device, dtype=torch.complex128)

        # define optimization function
        def func_op(x):
            return get_m_1_diag_vector(
                f_re_pe_ch_t=x, v_s=vvs, v_c=vvc, aha=batch_aha,
                lambda_c=lambda_c, lambda_s=lambda_s, indices=indices
            )

        # iter_bar = tqdm.trange(1, desc="Slice::Processing::Batch")
        xmin, res_vec, results = cgd(
            func_operator=func_op,
            x=torch.zeros_like(batch_k_space_x_y_ch_t, device=device), b=batch_k_space_x_y_ch_t.to(device),
            max_num_iter=max_num_iter,
            conv_tol=conv_tol,
            iter_bar=iter_bar
        )

        k_space_x_y_ch_t[:, :, :, idxs_echoes[start:end]] = xmin.cpu()

    # move read dir to back
    k_space_x_y_ch_t = torch.movedim(k_space_x_y_z_ch_t, 0, read_dir)
    return k_space_x_y_ch_t[:, :, None]


def get_m_1_diag_vector(
        f_re_pe_ch_t: torch.Tensor, aha: torch.Tensor, lambda_c: float, lambda_s: float, indices: torch.Tensor,
        v_s: torch.Tensor = torch.zeros((1, 1)), v_c: torch.Tensor = torch.zeros((1, 1))) -> torch.Tensor:
    """
    We define the M_1 matrix from A^H A f and the loraks_arxv operator P_x(f) V V^H,
    after extracting V from ACS data and getting the P_x based on the Loraks mode used.
    """
    # if input is zero vector -> S and C matrices will be 0 matrices. hence all multiplications will return 0
    # can skip compute aka matrix generation
    shape = f_re_pe_ch_t.shape
    if torch.max(torch.abs(f_re_pe_ch_t)) < 1e-7:
        return torch.zeros_like(f_re_pe_ch_t)
    m1_fhf = aha * f_re_pe_ch_t
    if lambda_c > 1e-6:
        m1_v_c = c_adjoint_operator(
            torch.matmul(
                c_operator(f_re_pe_ch_t, indices=indices),
                v_c
            ),
            indices=indices, k_space_dims=shape
        )
    else:
        m1_v_c = 0.0
    if lambda_s > 1e-6:
        m1_v_s = s_adjoint_operator(
            torch.matmul(
                s_operator(f_re_pe_ch_t, indices=indices),
                v_s
            ),
            indices=indices, k_space_dims=shape
        )
    else:
        m1_v_s = 0.0
    return m1_fhf - lambda_s * m1_v_s - lambda_c * m1_v_c

