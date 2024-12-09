import logging
import pathlib as plib

import torch
import numpy as np
import tqdm

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft, root_sum_of_squares
import plotly.graph_objects as go
import plotly.subplots as psub
from pymritools.utils import get_idx_2d_circular_neighborhood_patches_in_shape
from pymritools.recon.loraks.operators import s_operator, s_adjoint_operator
from pymritools.recon.loraks.algorithms import get_count_matrix
from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd


def main():
    # set output path
    path_out = plib.Path("./dev_sim/loraks").absolute()
    path_out.mkdir(exist_ok=True, parents=True)

    # set up  phantom
    shape = (256, 256)
    num_coils = 4
    num_echoes = 3
    logging.info("get SheppLogan phantom")
    logging.info("add virtual coils")
    sl_us_k = SheppLogan().get_sub_sampled_k_space(shape=shape, num_coils=num_coils, acceleration=3)

    logging.info("add decay echoes")
    # create decay echo images
    sl_us_k = sl_us_k[:, :, :, None] * torch.exp(-20 * torch.arange(1, 1 + num_echoes) * 0.009)[None, None, None, :]

    # to image space
    sl_us_img = fft(sl_us_k, img_to_k=True, axes=(0, 1))
    shape = sl_us_k.shape
    sl_us_img = root_sum_of_squares(sl_us_img, dim_channel=-2)
    # plot
    fig = psub.make_subplots(rows=1, cols=2)
    fig.add_trace(
        go.Heatmap(z=torch.abs(sl_us_img[:, :, 0]).numpy(), showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=torch.log(torch.abs(sl_us_k[:, :, 0, 0])).numpy(), showscale=False),
        row=1, col=2
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig_name = path_out.joinpath("naive_recon_us_rsos").with_suffix(".html")
    logging.info(f"Write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    sampling_mask = torch.abs(sl_us_k) > 1e-9
    # get actual samples
    sampled_k_us = sl_us_k[sampling_mask]

    # use loraks
    radius = 3
    rank = 50
    lam_s = 0.05
    max_num_iter = 15
    # __ One Time Calculations __
    # get dimensions
    shape = sl_us_k.shape
    n_read, n_phase, n_channels, n_echoes = shape

    # get indices for operators
    indices = get_idx_2d_circular_neighborhood_patches_in_shape(
        shape_2d=(n_read, n_phase), nb_radius=radius, device=torch.device("cpu")
    )
    # only use S matrix for now - only calculate for relevant dims
    count_matrix = get_count_matrix(shape=(n_read, n_phase, n_channels, n_echoes), indices=indices, mode="s")
    mask = count_matrix > 1e-7

    fig = psub.make_subplots(rows=num_echoes, cols=num_coils)
    for c in range(num_coils):
        for e in range(num_echoes):
            fig.add_trace(
                go.Heatmap(z=count_matrix[:, :, c, e], showscale=False),
                row=1+e, col=c+1
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig_name = path_out.joinpath("count_matrix").with_suffix(".html")
    logging.info(f"Write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    # calculate matrix dimensions
    m = (n_read - radius) * (n_phase - radius)
    n = (num_echoes * n_channels * indices.shape[1] * 2)

    matrix_rank = min(m, n)
    # build s_threshold based on rank
    s_threshold = torch.ones(matrix_rank, dtype=torch.float32)
    s_threshold[rank:] = 0

    # allocate result
    k = sl_us_k.clone().requires_grad_(True)

    # log losses
    losses = []
    bar = tqdm.trange(max_num_iter, desc="Optimization")
    for i in bar:
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
        loss_2 = torch.linalg.norm(k_recon_loraks[sampling_mask] - sampled_k_us)

        loss = loss_2 + lam_s * loss_1

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

        p_k = k[:, :, 0, 0].clone().detach()
        img = fft(p_k, axes=(0, 1), img_to_k=False)
        img_tmp = fft(k_recon_loraks[:, :, 0, 0].detach(), axes=(0, 1), img_to_k=False)
        grads = torch.abs(k.grad.detach()[:, :, 0, 0])
        fig = psub.make_subplots(
            rows=1, cols=4
        )
        fig.add_trace(
            go.Heatmap(z=torch.abs(img).numpy(), showscale=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=torch.log(torch.abs(p_k)).numpy(), showscale=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Heatmap(z=torch.abs(grads).numpy(), showscale=False),
            row=1, col=3
        )
        fig.add_trace(
            go.Heatmap(z=torch.abs(img_tmp).numpy(), showscale=False),
            row=1, col=4
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig_name = path_out.joinpath(f"k_candidate_iter-{i}").with_suffix(".html")
        logging.info(f"Write file: {fig_name}")
        fig.write_html(fig_name.as_posix())

    recon_k = k.detach()
    recon_img = fft(recon_k, img_to_k=False, axes=(0, 1))
    recon_img = root_sum_of_squares(recon_img, dim_channel=-2)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    main()
