import logging
import pathlib as plib
from typing import Union

import tqdm
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.recon.loraks.operators import (
    c_operator, c_adjoint_operator,
    s_operator, s_adjoint_operator
)
from pymritools.utils import get_idx_2d_circular_neighborhood_patches_in_shape, \
    get_idx_2d_square_neighborhood_patches_in_shape
from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd
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
    in_ones = torch.ones(shape, device=torch.device("cpu"), dtype=torch.complex64)
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
        ones_matrix,
        indices=indices, k_space_dims=shape
    ).real.to(torch.int)
    return count_matrix


def get_ac_region_from_sampling_pattern(sampling_mask_x_y_t: torch.Tensor):
    """
    Deduce the ac region in image space from the sampling pattern.
    Assume this region is centered on the image space.
    Assume this region is equal across all echoes.
    """
    # get dimensions
    nx, ny, nt = sampling_mask_x_y_t.shape
    # get center point
    nx_c, ny_c = nx // 2, ny // 2
    # init bounds assume at least 2 x 2 pixels (otherwise we would get None in first pass)
    top = ny_c
    bottom = ny_c - 1
    left = nx_c - 1
    right = nx_c

    # Expand region from center outward, both directions simultaneously
    for idx in range(max(nx_c, ny_c) - 1):
        if idx < nx_c:
            # check if where still within axis dims
            if torch.sum(sampling_mask_x_y_t[left-1:right+1, bottom:top], dim=-1).min() == nt:
                left -= 1
                right += 1
        if idx < ny_c:
            if torch.sum(sampling_mask_x_y_t[left:right, bottom-1:top+1], dim=-1).min() == nt:
                bottom -= 1
                top += 1

    shape = (right - left + 1, top - bottom + 1)
    start = torch.tensor([left, bottom])
    return start, shape


def get_v_matrix_of_ac_subspace(
        k_space_x_y_ch_t: torch.Tensor, ac_indices: torch.Tensor, mode: str,
        rank: int, compute_mode: str = "eigh"
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
    # m_ac = op(k_space_x_y_ch_t=k_space_x_y_ch_t, indices=indices)[ac_indices]
    m_ac = op(k_space_x_y_ch_t=k_space_x_y_ch_t, indices=ac_indices)
    if compute_mode == "eigh":
        # via eigh
        eig_vals, eig_vecs = torch.linalg.eigh(torch.matmul(m_ac.T, m_ac))
        m_ac_rank = eig_vals.shape[-1]
        # get subspaces from svd of subspace matrix
        eig_vals, idxs = torch.sort(torch.abs(eig_vals), descending=True)
        # eig_vecs_r = eig_vecs[idxs]
        eig_vecs = eig_vecs[:, idxs]
        # v_sub_r = eig_vecs_r[:self.rank].to(self.device)
        v_sub = eig_vecs[:, :rank]
    elif compute_mode == "svd":
        _, s, v = torch.linalg.svd(m_ac, full_matrices=False)
        m_ac_rank = s.shape[-1]
        v_sub = v[:, :rank]
    elif compute_mode == "rsvd":
        m_ac_rank = min(m_ac.shape[-2:])
        _, _, v = randomized_svd(matrix=m_ac, q=rank+10, power_projections=2)
        v_sub = v[:rank].conj().T
    elif compute_mode == "sor-svd":
        m_ac_rank = min(m_ac.shape[-2:])
        _, _, v = subspace_orbit_randomized_svd(matrix=m_ac, q=rank+10, power_projections=2)
        v_sub = v[:rank].conj().T
    elif compute_mode == "torch-lr":
        m_ac_rank = min(m_ac.shape[-2:])
        _, _, v = torch.svd_lowrank(A=m_ac, q=rank+10, niter=2)
        v_sub = v[:, :rank]
    else:
        err = f"compute mode {compute_mode} not implemented for AC subspace extraction"
        log_module.error(err)
        raise ValueError(err)
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
        data_consistency: float = 0.95,
        device: torch.device = torch.get_default_device()):
    # __ One Time Calculations __
    # get dimensions
    shape = k_space_x_y_z_ch_t.shape
    n_read, n_phase, n_slice, n_channels, n_echoes = shape
    # for now just pick middle slice
    idx_slice = int(n_slice / 2)
    k_space_x_y_ch_t = k_space_x_y_z_ch_t[:, :, idx_slice]
    slice_shape = k_space_x_y_ch_t.shape

    # get indices for operators
    indices = get_idx_2d_circular_neighborhood_patches_in_shape(
        shape_2d=(n_read, n_phase), nb_radius=radius, device=torch.device("cpu")
    )

    # only use S matrix for now - only calculate for relevant dims
    count_matrix = get_count_matrix(shape=(n_read, n_phase, n_channels, n_echoes), indices=indices, mode="s")
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
            s_matrix=matrix_recon_loraks, indices=indices, k_space_dims=slice_shape
        )
        k_recon_loraks[mask] /= count_matrix[mask]

        # take difference to sampled k for samples
        loss_2 = torch.linalg.norm(k_recon_loraks * sampling_mask_x_y_t[:, :, None] - k_space_x_y_ch_t)

        loss = data_consistency * loss_2 + (1 - data_consistency) * loss_1

        loss.backward()

        with torch.no_grad():
            k -= k.grad * 0.05

        k.grad.zero_()
        # optim.step()
        # optim.zero_grad()
        losses.append(loss.item())

        bar.postfix = (
            f"loss 1: {loss_1.item():.2f} -- loss 2: {loss_2.item():.2f} -- "
            f"total_loss: {loss.item():.2f} -- rank: {rank}"
        )

    return k[:, :, None]


def log_mem(point: str, device: torch.device):
    if not device.type == "cpu":
        logging.debug(f"Memory log: {point}")

        logging.debug(f"\t\t-{torch.cuda.get_device_name(device)}")
        logging.debug(f"\t\tAllocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1):.1f}, GB")
        logging.debug(f"\t\tCached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1):.1f}, GB")

        mem = torch.cuda.mem_get_info(device)
        logging.debug(f"\t\tAvailable: {mem[0] / 1024 ** 3:.1f} GB / {mem[1] / 1024 ** 3:.1f} GB")


def ac_loraks(
        k_space_x_y_z_ch_t: torch.Tensor,
        sampling_mask_x_y_t: torch.Tensor,
        radius: int,
        rank_c: int, lambda_c: float,
        rank_s: int, lambda_s: float,
        batch_size_channels: int = 16,
        max_num_iter: int = 10, conv_tol: float = 1e-3,
        visualize: bool = True, path_visuals: Union[str, plib.Path] = "",
        device: torch.device = torch.get_default_device()):
    # __ One Time Calculations __
    log_mem(point="AC Loraks Start", device=device)
    img_ac_start, img_ac_shape = get_ac_region_from_sampling_pattern(sampling_mask_x_y_t=sampling_mask_x_y_t)
    # find the indices to build LORAKS matrices for this particular shape
    # img_ac_indices = get_idx_2d_square_neighborhood_patches_in_shape(
    #     nb_shift=(1, 1),
    #     shape_2d=img_ac_shape, nb_size=radius+1, device=torch.device("cpu")
    # )
    # # add the starting points of the AC region rectangle to the indices
    # img_ac_indices += img_ac_start[None, None]

    # get dimensions
    k_space_x_y_z_ch_t = k_space_x_y_z_ch_t.to(torch.complex64)
    shape = k_space_x_y_z_ch_t.shape
    n_read, n_phase, n_slice, n_channels, n_echoes = shape
    # for now just pick middle slice
    # idx_slice = int(n_slice / 2)
    # k_space_x_y_ch_t = k_space_x_y_z_ch_t[:, :, idx_slice]

    # get indices for operators on whole image to calculate the count matrix for reconstruction
    indices = get_idx_2d_square_neighborhood_patches_in_shape(
        shape_2d=(n_read, n_phase), nb_size=2+radius, device=torch.device("cpu")
    )
    # get the indices within the ac region
    # convert to tensors
    img_ac_start = torch.tensor(img_ac_start, device=indices.device)  # Starting point (2,)
    img_ac_end = img_ac_start + torch.tensor(img_ac_shape, device=indices.device)  # End point (2,)

    inside_box = (indices > img_ac_start) & (indices < img_ac_end)  # Shape: (nxy, nb, 2)

    # Ensure all indices in the `nb` dimension satisfy the condition
    valid_indices_mask = inside_box.all(dim=-1).all(dim=1)  # Shape: (nxy,)

    # Extract valid indices using the mask
    img_ac_indices = indices[valid_indices_mask]  # Shape: (number_of_valid, nb, 2)

    plot_list = []
    plot_names = []
    # check if C matrix is used and calculate count matrix
    if lambda_c > 1e-7:
        c_count_matrix = get_count_matrix(
            mode="c", indices=indices, shape=(n_read, n_phase, 1, n_echoes)
        )
        plot_list.append(c_count_matrix)
        plot_names.append("C")
    else:
        c_count_matrix = 0

    # check if S matrix is used and calculate count matrix
    if lambda_s > 1e-7:
        s_count_matrix = get_count_matrix(
            mode="s", indices=indices, shape=(n_read, n_phase, 1, n_echoes)
        )
        plot_list.append(s_count_matrix)
        plot_names.append("S")
    else:
        s_count_matrix = 0

    aha = sampling_mask_x_y_t[:, :, None].to(torch.int) + lambda_c * c_count_matrix + lambda_s * s_count_matrix
    log_mem(point="End one time calculations", device=device)

    if visualize:
        # plot count matrices
        fig = psub.make_subplots(rows=2, cols=len(plot_list), subplot_titles=plot_names)
        for idx_i, i in enumerate(plot_list):
            fig.add_trace(
                go.Heatmap(z=torch.abs(i)[:, :, 0, 0], showscale=False),
                row=1, col=1 + idx_i
            )
        ac_region_plot = torch.zeros((n_read, n_phase))
        ac_region_plot[
            img_ac_start[0]:img_ac_start[0]+img_ac_shape[0],
            img_ac_start[1]:img_ac_start[1]+img_ac_shape[1]
        ] = 1

        fig.add_trace(
            go.Heatmap(z=ac_region_plot, showscale=False),
            row=2, col=1
        )
        if not path_visuals:
            path_visuals = plib.Path("./examples/recon/loraks")
        path_visuals = plib.Path(path_visuals).absolute()
        path_visuals.mkdir(parents=True, exist_ok=True)
        fig_name = path_visuals.joinpath('count-matrices_ac-region').with_suffix('.html')
        log_module.info(f"write file: {fig_name}")
        fig.write_html(fig_name.as_posix())

    for idx_s in range(n_slice):
        log_module.info(f"Processing slice :: {idx_s+1} / {n_slice}")
        log_mem(point=f"Processing slice :: Start :: {idx_s+1} / {n_slice}", device=device)
        # __ need to batch. we can just permute and batch channels
        # ToDo: implement multiple random permutations and average?
        #  This way we are less prone to batching non sensitive channels for reconstruction?
        #  Alternatively we could do a fft us recon, extract some crude sensitivity maps and
        #  batch channels correlated in image space
        # idxs_channels = torch.randperm(n_channels)
        num_batches = int(np.ceil(n_channels / batch_size_channels))
        iter_bar = tqdm.trange(num_batches, desc="batch_processing") if visualize else range(num_batches)
        for idx_b in iter_bar:
            log_module.debug(f"Processing batch :: {idx_b+1} / {num_batches}")
            log_mem(point=f"Processing batch :: {idx_b+1} / {num_batches}", device=device)
            start = idx_b * batch_size_channels
            end = np.min([(idx_b + 1) * batch_size_channels, n_channels])
            batch_k_space_x_y_ch_t = k_space_x_y_z_ch_t[:, :, idx_s, start:end].to(device)
            batch_aha = aha[:, :, start:end].to(device)

            # __ per slice calculations
            # __ for C
            if lambda_c > 1e-9:
                vvc = get_v_matrix_of_ac_subspace(
                    k_space_x_y_ch_t=batch_k_space_x_y_ch_t, compute_mode="torch-lr",
                    ac_indices=img_ac_indices, mode="c", rank=rank_c
                )
            else:
                vvc = torch.zeros(1, device=device, dtype=batch_k_space_x_y_ch_t.dtype)

            # __ for S
            if lambda_s > 1e-9:
                vvs = get_v_matrix_of_ac_subspace(
                    k_space_x_y_ch_t=batch_k_space_x_y_ch_t, compute_mode="torch-lr",
                    ac_indices=img_ac_indices, mode="s", rank=rank_s
                )
            else:
                vvs = torch.zeros(1, device=device, dtype=batch_k_space_x_y_ch_t.dtype)

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

            k_space_x_y_z_ch_t[:, :, idx_s, start:end] = xmin.cpu()
        log_mem(point=f"Processing slice :: End :: {idx_s+1} / {n_slice}", device=device)

    return k_space_x_y_z_ch_t


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

