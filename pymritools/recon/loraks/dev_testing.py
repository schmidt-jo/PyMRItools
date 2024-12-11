import logging
import pathlib as plib

import torch
from torch.optim import Adam
import numpy as np
import tqdm
import polars as pl

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft, root_sum_of_squares
import plotly.graph_objects as go
import plotly.subplots as psub
from pymritools.utils import get_idx_2d_circular_neighborhood_patches_in_shape
from pymritools.recon.loraks.operators import s_operator, s_adjoint_operator
from pymritools.recon.loraks.algorithms import get_count_matrix
from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd


def func_optim(k, indices, s_threshold, shape, count_matrix, mask, sl_us_k, sampling_mask, lam_s):
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
    # if not matrix_space:
    k_recon_loraks = s_adjoint_operator(
        s_matrix=matrix, indices=indices, k_space_dims=shape
    )
    k_recon_loraks[mask] /= count_matrix[mask]
    loss_2 = torch.linalg.norm(k_recon_loraks[sampling_mask] - sl_us_k[sampling_mask])
    # else:
    #     # take difference to sampled k for samples
    #     loss_2 = torch.linalg.norm(matrix * sampling_mask_matrix_space - matrix_us_k)

    return loss_2 + lam_s * loss_1, loss_1, loss_2

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

    # set LORAKS parameters
    radius = 3
    rank = 40
    lam_s = 0.05
    max_num_iter = 100
    device = torch.device("cuda")

    # lr = np.linspace(0.1, 0.005, max_num_iter)
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

    # calculate matrix dimensions
    m = (n_read - radius) * (n_phase - radius)
    n = (n_echoes * n_channels * indices.shape[1] * 2)
    matrix_rank = min(m, n)

    # embed image in random image to ensure svd gradient stability
    k_init = torch.randn_like(sl_us_k)
    k_init *= 0.01 * torch.max(torch.abs(sl_us_k)) / torch.max(torch.abs(k_init))
    k_init[sampling_mask] = sl_us_k[sampling_mask]
    # want to multiply search candidates by sampling mask to compute data consistency term, cast to numbers
    sampling_mask = sampling_mask.to(device=device)
    # save us data for consistency term, send to device
    sl_us_k = sl_us_k.to(device)

    # build s_threshold based on rank
    s_threshold = torch.ones(matrix_rank, dtype=torch.float32)
    s_threshold[rank:] = 0
    s_threshold = s_threshold.to(device)

    # allocate result
    k_init = k_init.to(device)
    k = k_init.clone().requires_grad_(True)

    # log losses
    losses = []
    # plot intermediates
    plot_k = [sl_us_k[:, :, 0, 0].cpu()]
    plot_img = [sl_us_img[:, :, 0].cpu()]
    plot_grads = [torch.zeros_like(plot_k[-1])]

    bar = tqdm.trange(max_num_iter, desc="Optimization")

    for i in bar:
        loss, loss_1, loss_2 = func_optim(
            k=k, indices=indices, s_threshold=s_threshold, shape=shape, count_matrix=count_matrix, mask=mask,
            sl_us_k=sl_us_k, sampling_mask=sampling_mask, lam_s=lam_s
        )
        loss.backward()

        with torch.no_grad():
            grads = k.grad
            # at this stage we have the gradients g based on the current best guess (initially g_0).
            # to minimize the number of steps to take, aka conjugate gradient method we do a few things.
            # 1) compute search direction (Polak-Ribiere formula): beta_k = g_{k+1}^T  (g_{k+1} - g_k) / (g_k^T g_k)
            # 6) update search direction: d_{k+1} = -g_{k+1} + beta_k d_k
            if i == 0:
                # setting initial search direction d_0 = g_0
                d_k = -grads
                g_k = grads
            else:
                # compute conjugate gradient direction
                # flatten the gradients
                g_k_flat = g_k.view(-1)
                g_k1_flat = grads.view(-1)
                nom = torch.dot(g_k1_flat, g_k1_flat - g_k_flat)
                denom = torch.dot(g_k_flat, g_k_flat)
                beta = nom / denom if torch.abs(denom) > 1e-9 else 0.0

                # update search direction
                d_k = -grads + beta * d_k
                g_k = grads

            # 2) line search: find optimal step size a_k that minimizes the function along this step direction d_k.
            # armijo sufficient decrease condition
            c_a = 1e-2
            alpha = 0.5
            max_num_iter_line_search = 100
            for iter_line_search in range(max_num_iter_line_search):
                a, _, _ = func_optim(
                    k=k + alpha * d_k, indices=indices, s_threshold=s_threshold, shape=shape,
                    count_matrix=count_matrix, mask=mask, sl_us_k=sl_us_k, sampling_mask=sampling_mask, lam_s=lam_s
                )
                b = loss + c_a * alpha * torch.linalg.norm(g_k * d_k)
                if a < b:
                    break
                alpha *= 0.4

            # 3) update position: move to x_{k+1} = x_{k} + a_k d_k
            k += alpha * d_k
            # 4) compute new gradient: g_{k+1} = nabla f(x_{k+1})
            #
            #

            grads = torch.abs(grads)
            conv = torch.linalg.norm(grads).cpu()
            grads = grads[:, :, 0, 0].cpu()

        k.grad.zero_()

        # optim.step()
        # optim.zero_grad()
        losses.append(
            {"total": loss.item(), "data": loss_2.item(), "low rank": loss_1.item(),
             "conv": conv.item(), "alpha": alpha}
        )

        bar.postfix = (
            f"loss low rank: {1e3*loss_1.item():.2f} -- loss data: {1e3*loss_2.item():.2f} -- "
            f"total_loss: {1e3*loss.item():.2f} -- conv : {conv.item()} -- alpha: {1e3*alpha} -- rank: {rank}"
        )

        if i in np.unique(np.logspace(0.1, np.log2(max_num_iter), 10, base=2, endpoint=True).astype(int)):
            # some plotting intermediates
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

    losses = pl.DataFrame(losses)

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(y=losses["total"], name="total loss", mode="lines")
    )
    fig.add_trace(
        go.Scattergl(y=losses["data"], name="data", mode="lines")
    )
    fig.add_trace(
        go.Scattergl(y=losses["low rank"], name="low rank", mode="lines")
    )
    fig.add_trace(
        go.Scattergl(y=losses["conv"], name="conv", mode="lines")
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
