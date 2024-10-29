import logging
import pathlib as plib

import tqdm
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.recon.loraks.algorithm.operators import (
    c_operator, c_adjoint_operator,
    s_operator, s_adjoint_operator
)
from pymritools.utils import get_idx_2d_circular_neighborhood_patches_in_shape
from pymritools.utils.algorithms import cgd

log_module = logging.getLogger(__name__)

def ac_loraks(
        k_space_x_y_z_ch_t: torch.Tensor,
        sampling_mask_x_y_t: torch.Tensor,
        radius: int,
        rank_c: int, lambda_c: float,
        rank_s: int, lambda_s: float,
        max_num_iter: int = 10, conv_tol: float = 1e-3,
        device: torch.device = torch.get_default_device()):
    # __ One Time Calculations __
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

    # want to extract count matrices - dont need to calculate for redundant dimensions
    c_ones_matrix = c_operator(
        k_space_x_y_ch_t=torch.ones(
            (n_read, n_phase, 1, n_echoes), dtype=torch.complex128, device=torch.device("cpu")
        ),
        indices=indices
    )
    c_count_matrix = c_adjoint_operator(
        c_matrix=c_ones_matrix,
        indices=indices, k_space_dims=(n_read, n_phase, 1, n_echoes)
    ).real.to(torch.int)

    s_ones_matrix = s_operator(
        k_space_x_y_ch_t=torch.ones(
            (n_read, n_phase, 1, n_echoes),
            device=torch.device("cpu"), dtype=torch.complex128
        ),
        indices=indices
    )
    s_count_matrix = s_adjoint_operator(
        s_matrix=s_ones_matrix,
        indices=indices, k_space_dims=(n_read, n_phase, 1, n_echoes)
    ).real.to(torch.int)

    aha = sampling_mask_x_y_t[:, :, None].to(torch.int) + lambda_c * c_count_matrix + lambda_s * s_count_matrix

    # want to extract ac indices
    # we build the matrix from the mask and then look for the neighborhoods which are fully contained for
    # all concatenated echoes / channels
    c_ac = sampling_mask_x_y_t[indices[..., 0], indices[..., 1]].to(torch.int)
    c_ac_idxs = torch.sum(c_ac, dim=(1, 2)) == c_ac.shape[1] * c_ac.shape[2]

    s_ac_m = torch.flip(sampling_mask_x_y_t, dims=(0, 1))[indices[..., 0], indices[..., 1]].to(torch.int)
    s_ac_idxs = (torch.sum(c_ac, dim=(1, 2)) + torch.sum(s_ac_m, dim=(1, 2))) == c_ac.shape[1] * c_ac.shape[2] * 2
    s_ac_idxs = torch.tile(s_ac_idxs, dims=(2,))

    # plot count matrices
    fig = psub.make_subplots(rows=2, cols=2, subplot_titles=('C', 'S'))
    for idx_i, i in enumerate([c_count_matrix, s_count_matrix]):
        fig.add_trace(
            go.Heatmap(z=torch.abs(i)[:, :, 0, 0], showscale=False),
            row=1, col=1 + idx_i
        )
    # want to translate the indices into an image
    c_ones_matrix[~c_ac_idxs] = 0
    c_ac_img = c_adjoint_operator(c_ones_matrix, indices=indices, k_space_dims=(n_read, n_phase, 1, n_echoes))
    s_ones_matrix[~s_ac_idxs] = 0
    s_ac_img = s_adjoint_operator(s_ones_matrix, indices=indices, k_space_dims=(n_read, n_phase, 1, n_echoes))
    for idx_i, i in enumerate([c_ac_img, s_ac_img]):
        fig.add_trace(
            go.Heatmap(z=torch.abs(i)[:, :, 0, 0], showscale=False),
            row=2, col=1 + idx_i
        )
    fig_name = plib.Path("./examples/recon/loraks").joinpath('count-matrices_ac-region').with_suffix('.html')
    log_module.info(f"write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    # __ need to batch. we can just permute and batch echoes
    num_batched_echoes = 3
    idxs_echoes = torch.randperm(n_echoes)
    num_batches = int(np.ceil(n_echoes / num_batched_echoes))

    iter_bar = tqdm.trange(num_batches, desc="batch_processing")
    for idx_b in iter_bar:
        start = idx_b * num_batched_echoes
        end = np.min([(idx_b + 1) * num_batched_echoes, n_echoes])
        batch_k_space_x_y_ch_t = k_space_x_y_ch_t[:, :, :, idxs_echoes[start:end]].to(device)
        batch_aha = aha[:, :, :, idxs_echoes[start:end]].to(device)

        # __ per slice calculations
        # ToDo: we know the ACS indices already, can we calculate this not with the full batch and masking afterwards,
        # but the other way around. i.e. masking the k_space batch with the found AC region indices and
        # then building c and s matrix?
        # __ for C
        # find the V matrix for the ac subspaces
        m_ac = c_operator(k_space_x_y_ch_t=batch_k_space_x_y_ch_t, indices=indices)[c_ac_idxs]
        # put on device if not there already
        m_ac = m_ac.to(device)
        # via eigh
        eig_vals, eig_vecs = torch.linalg.eigh(torch.matmul(m_ac.T, m_ac))
        m_ac_rank = eig_vals.shape[0]
        # get subspaces from svd of subspace matrix
        eig_vals, idxs = torch.sort(torch.abs(eig_vals), descending=True)
        # eig_vecs_r = eig_vecs[idxs]
        eig_vecs = eig_vecs[:, idxs]
        # v_sub_r = eig_vecs_r[:self.rank].to(self.device)
        v_sub_c = eig_vecs[:, :rank_c].to(device)
        if m_ac_rank < rank_c:
            err = f"loraks rank parameter is too large, cant be bigger than ac matrix dimensions."
            log_module.error(err)
            raise ValueError(err)

        # __ for S
        # find the V matrix for the ac subspaces
        m_ac = s_operator(k_space_x_y_ch_t=batch_k_space_x_y_ch_t, indices=indices)[s_ac_idxs]
        # put on device if not there already
        m_ac = m_ac.to(device)
        # via eigh
        eig_vals, eig_vecs = torch.linalg.eigh(torch.matmul(m_ac.T, m_ac))
        m_ac_rank = eig_vals.shape[0]
        # get subspaces from svd of subspace matrix
        eig_vals, idxs = torch.sort(torch.abs(eig_vals), descending=True)
        # eig_vecs_r = eig_vecs[idxs]
        eig_vecs = eig_vecs[:, idxs]
        # v_sub_r = eig_vecs_r[:self.rank].to(self.device)
        v_sub_s = eig_vecs[:, :rank_s].to(device)
        if m_ac_rank < rank_s:
            err = f"loraks rank parameter is too large, cant be bigger than ac matrix dimensions."
            log_module.error(err)
            raise ValueError(err)

        vvs = torch.matmul(v_sub_s, v_sub_s.conj().T)
        vvc = torch.matmul(v_sub_c, v_sub_c.conj().T)

        # defince optimization function
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
    We define the M_1 matrix from A^H A f and the loraks operator P_x(f) V V^H,
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


if __name__ == '__main__':
    # ac_loraks()
    pass
