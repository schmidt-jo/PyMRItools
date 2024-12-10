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

    sampling_mask = (torch.abs(sl_us_k) > 1e-9)

    # use loraks
    radius = 3
    rank = 40
    lam_s = 0.05
    max_num_iter = 100
    device = torch.device("cuda")
    matrix_space = False
    lr = np.linspace(0.1, 0.005, max_num_iter)
    # __ One Time Calculations __
    # get dimensions
    shape = sl_us_k.shape
    n_read, n_phase, n_channels, n_echoes = shape

    # get indices for operators
    indices = get_idx_2d_circular_neighborhood_patches_in_shape(
        shape_2d=(n_read, n_phase), nb_radius=radius, device=torch.device("cpu")
    )
    # only use S matrix for now - only calculate for relevant dims
    count_matrix = get_count_matrix(shape=(n_read, n_phase, n_channels, n_echoes), indices=indices, mode="s").to(device)
    mask = count_matrix > 1e-7

    fig = psub.make_subplots(rows=num_echoes, cols=num_coils)
    for c in range(num_coils):
        for e in range(num_echoes):
            fig.add_trace(
                go.Heatmap(z=count_matrix[:, :, c, e].cpu(), showscale=False),
                row=1+e, col=c+1
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig_name = path_out.joinpath("count_matrix").with_suffix(".html")
    logging.info(f"Write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    if not matrix_space:
        m = (n_read - radius) * (n_phase - radius)
        n = (num_echoes * n_channels * indices.shape[1] * 2)
        matrix_rank = min(m, n)
        # embed image in random image to ensure svd gradient stability
        k_init = torch.randn_like(sl_us_k)
        k_init *= 0.01 * torch.max(torch.abs(sl_us_k)) / torch.max(torch.abs(k_init))
        k_init[sampling_mask] = sl_us_k[sampling_mask]
        sampling_mask = sampling_mask.to(device=device)
        sl_us_k = sl_us_k.to(device)

    else:
        matrix_us_k = s_operator(
            k_space_x_y_ch_t=sl_us_k, indices=indices
        ).to(device)
        matrix_rank = np.min([*matrix_us_k.shape])
        sampling_mask_matrix_space = torch.abs(matrix_us_k) > 1e-9
        # embedd matrix in non-zero init, to prevent ill defined svd
        matrix_init = torch.randn_like(matrix_us_k)
        matrix_init *= 0.2 * torch.max(torch.abs(matrix_us_k)) / torch.max(torch.abs(matrix_init))
        matrix_init[sampling_mask_matrix_space] = matrix_us_k[sampling_mask_matrix_space]
        sampling_mask_matrix_space = sampling_mask_matrix_space.to(dtype=torch.int, device=device)

    # build s_threshold based on rank
    s_threshold = torch.ones(matrix_rank, dtype=torch.float32)
    s_threshold[rank:] = 0
    s_threshold = s_threshold.to(device)

    # allocate result
    if not matrix_space:
        k_init = k_init.to(device)
        k = k_init.clone().requires_grad_(True)
    else:
        matrix = matrix_init.clone().requires_grad_(True).to(device)

    # log losses
    losses = []
    # plot intermediates
    plot_k = [sl_us_k[:, :, 0, 0].cpu()]
    plot_img = [sl_us_img[:, :, 0].cpu()]
    plot_grads = [torch.zeros_like(plot_k[-1])]

    bar = tqdm.trange(max_num_iter, desc="Optimization")

    for i in bar:
        if not matrix_space:
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
        if not matrix_space:
            k_recon_loraks = s_adjoint_operator(
                s_matrix=matrix, indices=indices, k_space_dims=shape
            )
            k_recon_loraks[mask] /= count_matrix[mask]
            loss_2 = torch.linalg.norm(k_recon_loraks[sampling_mask] - sl_us_k[sampling_mask])
        else:
            # take difference to sampled k for samples
            loss_2 = torch.linalg.norm(matrix * sampling_mask_matrix_space - matrix_us_k)

        loss = loss_2 + lam_s * loss_1

        loss.backward()

        # at this stage we have the gradients g based on the current best guess (initially g_0).
        # to minimize the number of steps to take, aka conjugate gradient method we do a few things.
        # 1) setting initial search direction d_0 = - g_0
        # 2) line search: find optimal step size a_k that minimizes the function along this step direction d_k.
        # 3) update position: move to x_{k+1} = x_{k} + a_k d_k
        # 4) compute new gradient: g_{k+1} = nabla f(x_{k+1})
        # 5) compute search direction (Polak-Ribiere formula): beta_k = g_{k+1}^T  (g_{k+1} - g_k) / (g_k^T g_k)
        # 6) update search direction: d_{k+1} = -g_{k+1} + beta_k d_k

        with torch.no_grad():
            if not matrix_space:
                k -= k.grad * lr[i]
                grad_img = k.grad.detach()
            else:
                matrix -= matrix.grad * lr[i]
                grad_img = s_adjoint_operator(matrix.grad.detach(), indices=indices, k_space_dims=shape)
            grads = torch.abs(grad_img[:, :, 0, 0]).cpu()

        k.grad.zero_()



        # optim.step()
        # optim.zero_grad()
        losses.append(loss.item())

        bar.postfix = (
            f"loss low rank: {loss_1.item():.2f} -- loss data: {loss_2.item():.2f} -- "
            f"total_loss: {loss.item():.2f} -- rank: {rank}"
        )

        if i in np.unique(np.logspace(0.1, np.log2(max_num_iter), 10, base=2, endpoint=True).astype(int)):
            # some plotting intermediates
            if matrix_space:
                k_recon_loraks = s_adjoint_operator(matrix.clone().detach(), indices=indices, k_space_dims=shape)
            else:
                k_recon_loraks = k.clone().detach()
            p_k = k_recon_loraks[:, :, :, 0].clone().detach().cpu()
            img = fft(p_k, axes=(0, 1), img_to_k=False).cpu()

            plot_k.append(p_k[:, :, 0])
            plot_img.append(root_sum_of_squares(img, dim_channel=-1))
            plot_grads.append(grads)

    fig = psub.make_subplots(
        rows=3, cols=len(plot_k),
    )
    for i, pk in enumerate(plot_k):
        fig.add_trace(
            go.Heatmap(z=torch.abs(plot_img[i]).numpy(), showscale=False),
            row=1, col=1+i
        )
        fig.add_trace(
            go.Heatmap(z=torch.log(torch.abs(pk)).numpy(), showscale=False),
            row=2, col=1+i
        )
        fig.add_trace(
            go.Heatmap(z=torch.abs(plot_grads[i]).numpy(), showscale=False),
            row=3, col=1+i
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig_name = path_out.joinpath(f"k_optim").with_suffix(".html")
    logging.info(f"Write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(y=losses)
    )
    fig_name = path_out.joinpath(f"losses").with_suffix(".html")
    logging.info(f"Write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    # recon_k = k.detach()
    # recon_img = fft(recon_k, img_to_k=False, axes=(0, 1))
    # recon_img = root_sum_of_squares(recon_img, dim_channel=-2)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    main()
