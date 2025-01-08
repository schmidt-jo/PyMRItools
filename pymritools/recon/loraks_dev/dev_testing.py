import logging
import pathlib as plib

import torch
import numpy as np
import tqdm
import polars as pl

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft, root_sum_of_squares
import plotly.graph_objects as go
import plotly.subplots as psub
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape, get_linear_indices
from pymritools.recon.loraks_dev.operators import s_operator, s_adjoint_operator, c_operator, c_adjoint_operator, s_operator_mem_opt
from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd, cgd, \
    subspace_orbit_randomized_svd


# ToDo: 2 step LORAKS process:
#   a) prepare k-space into batch dimensions and rest - with complementary "backtransform"
#   b) actual Loraks algorithm, working on the batched shape
# ToDo jochen: 3 easy testcases (no channels, w channels, w echoes & channels, into matlab, and get the datasets back to torch for comparsiong


# a)
def prepare_k_shape(k_space_input: torch.Tensor) -> torch.Tensor:
    return torch.movedim(k_space_input, 2, 0)


def reshape_k_shape(k_shape: torch.Tensor) -> torch.Tensor:
    return torch.movedim(k_shape, 0, 2)


def func_optim(k, indices, s_threshold, k_sampled_points, sampling_mask, lam_s, q, matrix_shape):
    # ToDo: make sure to have operators "true" linear indexing ready
    # use matrix indexing -> reshape / view in 2D [dim_sampling, dim_neighborhood]
    # get operator matrix
    # matrix = s_operator(
    #     k_space=k, indices=indices
    # )

    # reshape to operator - for c operator
    matrix = k.view(-1)[indices].view(matrix_shape)

    # do svd
    # we can use torch svd, or try the randomized versions
    # u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    # u, s, vh = randomized_svd(matrix=matrix, q=rq, power_projections=2)
    # u, s, vh = subspace_orbit_randomized_svd(matrix=matrix, q=q, power_projections=2)
    u, s, vh = torch.svd_lowrank(A=matrix, q=q, niter=2)

    # threshold singular values
    s_r = s * s_threshold

    # reconstruct the low rank approximation
    # u * s @ vh
    matrix_recon_loraks = torch.matmul(u * s_r.to(u.dtype), vh.mH)

    # __ Low Rank Loss
    # enforce low rank matrix structure by minimizing distance of LORAKS matrix to its truncated SVD
    loss_1 = torch.linalg.norm(matrix - matrix_recon_loraks, ord="fro")

    # __ Data Loss
    # enforce data - consistency by minimizing distance to known sampled points
    loss_2 = torch.linalg.norm(k[sampling_mask] - k_sampled_points)
    return loss_2 + lam_s * loss_1, loss_1, loss_2


def func_optim_js(k, indices, s_threshold, k_sampled_points, sampling_mask, lam_s, q):
    # ToDo: make sure to have operators "true" linear indexing ready
    # use matrix indexing -> reshape / view in 2D [dim_sampling, dim_neighborhood]
    # get operator matrix
    matrix = c_operator(
        k_space=k, indices=indices
    )
    # matrix = s_operator_mem_opt(
    #     k_space=k, indices=indices, matrix_shape=matrix_shape
    # )

    # reshape to operator - for c operator
    # matrix = k.view(-1)[indices].view(matrix_shape)

    # do svd
    # we can use torch svd, or try the randomized versions
    # u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    # u, s, vh = randomized_svd(matrix=matrix, q=rq, power_projections=2)
    # u, s, vh = subspace_orbit_randomized_svd(matrix=matrix, q=q, power_projections=2)
    u, s, vh = torch.svd_lowrank(A=matrix, q=q, niter=2)
    # TODO: Jochen, use q=rank+2 or something if you want, but you need to

    # threshold singular values
    s_r = s * s_threshold

    # reconstruct the low rank approximation
    # u * s @ vh
    matrix_recon_loraks = torch.matmul(u * s_r.to(u.dtype), vh.mH)

    # __ Low Rank Loss
    # enforce low rank matrix structure by minimizing distance of LORAKS matrix to its truncated SVD
    loss_1 = torch.linalg.norm(matrix - matrix_recon_loraks, ord="fro")

    # __ Data Loss
    # enforce data - consistency by minimizing distance to known sampled points
    loss_2 = torch.linalg.norm(k[sampling_mask] - k_sampled_points)

    return loss_2 + lam_s * loss_1, loss_1, loss_2


def create_phantom(nx: int = 256, ny: int = 256, nc: int = 1, ne: int = 1):
    # set up  phantom
    shape = (nx, ny)
    logging.info("get SheppLogan phantom")
    logging.info("add virtual coils")
    sl_us_k = SheppLogan().get_sub_sampled_k_space(
        shape=shape, num_coils=nc, acceleration=2, num_echoes=ne
    )
    sl_us_k = torch.flip(sl_us_k, dims=(0,))

    logging.info("add misc dimension and set input shape")
    # We want to set LORAKS input dimensions to be (nx, ny, nz, nc, ne, m)
    # hence insert slice dim
    sl_us_k = sl_us_k.unsqueeze(2)
    # insert any dims at end that are not provided upon creation process
    if nc is None or nc == 1:
        sl_us_k = sl_us_k.unsqueeze(3)
    if ne is None or ne == 1:
        sl_us_k = sl_us_k.unsqueeze(4)
    sl_us_k = sl_us_k.unsqueeze(-1)

    # plot
    img = fft(sl_us_k, axes=(0, 1))
    fig = psub.make_subplots(
        cols=2 * ne, rows=nc,
        row_titles=[f"Coil: {k + 1}" for k in range(nc)],
        vertical_spacing=0.02, horizontal_spacing=0.02
    )
    for e in range(ne):
        fig.add_annotation(
            x=(1 / 2 + e) / ne, y=1.05, xref="paper", yref="paper", font=dict(size=16),
            xanchor="center", yanchor="middle", text=f"Echo: {e+1}", showarrow=False,
        )
        for c in range(nc):
            fig.add_trace(
                go.Heatmap(
                    z=torch.squeeze(torch.abs(img[:, :, :, c, e])),
                    colorscale="Gray", showscale=False
                ), col=1 + 2 * e, row=1 + c
            )
            xaxis = fig.data[-1].xaxis
            fig.update_yaxes(visible=False, scaleanchor=xaxis, scaleratio=1, col=1 + 2 * e, row=1 + c)

            fig.add_trace(
                go.Heatmap(
                    z=torch.squeeze(torch.log(torch.abs(sl_us_k[:, :, :, c, e]))),
                    showscale=False
                ), col=2 * (e + 1), row=1 + c
            )
            xaxis = fig.data[-1].xaxis
            fig.update_yaxes(visible=False, scaleanchor=xaxis, scaleratio=1, col=2 * (e + 1), row=1 + c)

    fig.update_xaxes(visible=False)
    fig_name = plib.Path("./dev_sim/loraks").absolute().joinpath(f"phantom").with_suffix(".html")
    logging.info(f"Write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    # get sampling mask in input shape
    sampling_mask = (torch.abs(sl_us_k) > 1e-9)

    # embed image in random image to ensure svd gradient stability
    k_init = torch.randn_like(sl_us_k)
    k_init *= 1e-3 * torch.max(torch.abs(sl_us_k)) / torch.max(torch.abs(k_init))
    k_init[sampling_mask] = sl_us_k[sampling_mask]

    return k_init.contiguous(), sampling_mask.contiguous()


def comparison_js(k_load: torch.Tensor, sampling_mask: torch.Tensor,
                  lr: torch.Tensor, device: torch.device,
                  q: int, max_num_iter: int, s_threshold: torch.Tensor, lam_s: float):
    load_shape = k_load.shape
    # get indices for operators - direction is along x and y, set those to 1
    indices_mapping, batch_reshape = get_all_idx_nd_square_patches_in_nd_shape(
        size=5, patch_direction=(1, 1, 0, 0, 0, 0), k_space_shape=k_load.shape,
        combination_direction=(0, 0, 0, 1, 1, 1)
    )
    # returns dims [b (non patch, non combination dims), n_patch, n_combination + neighborhood]
    # need to reshape the input k-space
    k_input = torch.reshape(k_load, batch_reshape)
    # indices_mapping, matrix_shape = get_linear_indices(
    #     k_space_shape=k_input.shape[1:], patch_shape=(5, 5, -1, -1, -1), sample_directions=(1, 1, 0, 0, 0)
    # )
    k_out = torch.zeros_like(k_input)
    sampling_mask = torch.reshape(sampling_mask, batch_reshape)
    # log losses and plot data
    losses = []
    #
    # # plot intermediates
    # sl_us_img = fft(k_input, img_to_k=False, axes=(0, 1))
    # sl_us_img = root_sum_of_squares(sl_us_img, dim_channel=-3)
    # plot_k = [k_input[:, :, 0, 0, 0, 0].cpu()]
    # plot_img = [sl_us_img[:, :, 0, 0, 0].cpu()]
    # plot_grads = [torch.zeros_like(plot_k[-1])]
    # plot_names = [0]

    # setup iterations
    bar = tqdm.trange(max_num_iter, desc="Optimization")

    for b in range(k_input.shape[0]):
        k = k_input[b].clone().to(device).requires_grad_(True)
        index_batch = indices_mapping[b].to(device)
        # index_batch = indices_mapping.to(device)
        sampling_mask_batch = sampling_mask[b].to(device)
        k_sampled_points = k_input[b].to(device)[sampling_mask_batch]

        for i in bar:
            loss, loss_1, loss_2 = func_optim_js(
                k=k, indices=index_batch, s_threshold=s_threshold, q=q,
                k_sampled_points=k_sampled_points, sampling_mask=sampling_mask_batch, lam_s=lam_s,
                # matrix_shape=matrix_shape
            )
            loss.backward()

            # if i == 0 or search_direction is None:
            #     # Zu Beginn wird die Suchrichtung als negativer Gradient initialisiert
            #     search_direction = -grad
            #     beta = 0  # Kein Beta-Wert beim ersten Schritt
            # else:
            #     # Beta berechnen
            #     beta = torch.dot(grad.flatten(), grad.flatten()) / torch.dot(prev_gradient.flatten(),
            #                                                                  prev_gradient.flatten())
            #     # Konjugierte Suchrichtung aktualisieren
            #     search_direction = -grad + beta * search_direction
            #
            # # Lernrate mit Armijo-Zeilensuche bestimmen
            # learning_rate = armijo_line_search(
            #     k=k, loss=loss, grad=grad, direction=search_direction, func_optim=func_optim,
            #     index_batch=index_batch, s_threshold=s_threshold, k_input_batch=k_sampled_points,
            #     sampling_mask_batch=sampling_mask_batch, lam_s=lam_s, rank=rank
            # )
            # learning_rate = armijo_search(loss_func=func, param=k, direction=search_direction, grad=grad)

            # Use the optimal learning_rate to update parameters
            with torch.no_grad():
                k -= lr[i] * k.grad
                # grads = torch.abs(k.grad)
                # conv = torch.linalg.norm(grads - prev_gradient)
                # prev_gradient = grad.clone()
                #
                # if i in np.unique(np.logspace(0.1, np.log2(max_num_iter), 10, base=2, endpoint=True).astype(int) - 1):
                #     # save for plotting on some iterations
                #     grads = grads[:, :, 0, 0].cpu()
                #
                #     # some plotting intermediates
                #     k_recon_loraks = k.clone().detach()
                #     p_k = k_recon_loraks[:, :, :, 0, 0].clone().detach().cpu()
                #     img = fft(p_k, axes=(0, 1), img_to_k=False).cpu()
                #
                #     plot_k.append(p_k[:, :, 0])
                #     plot_img.append(root_sum_of_squares(img, dim_channel=-1))
                #     plot_grads.append(grads[:, :, 0])
                #     plot_names.append(i + 1)

            # if i % 2 == 0:
            k.grad.zero_()
            bar.postfix = (
                f"loss low rank: {1e3 * loss_1.item():.2f} -- loss data: {1e3 * loss_2.item():.2f} -- "
                f"total_loss: {1e3 * loss.item():.2f}"
            )
        k_out[b] = k.detach().cpu()
    return torch.reshape(k_out, load_shape), k_input


def core(
        k_load: torch.Tensor, sampling_mask: torch.Tensor, lr: torch.Tensor,
        device: torch.device, q: int, max_num_iter: int, s_threshold: torch.Tensor, lam_s: float):
    # a) shaping -> dims # dims [batch, dim_a, ... dim_n]  -> loraks matrices build from dim_a, ..., dim_n
    # -> should fit on GPU
    k_input = prepare_k_shape(k_load)
    sampling_mask = prepare_k_shape(k_space_input=sampling_mask)
    batch, *dims = k_input.shape
    k_out = torch.zeros_like(k_input)

    # b) indexing
    patch_shape = (5, 5, -1, -1, -1)
    sampling_directions = (1, 1, 0, 0, 0)

    indices, loraks_matrix_shape = get_linear_indices(
        k_space_shape=dims, patch_shape=patch_shape, sample_directions=sampling_directions
    )
    # does it make a difference to put them on GPU?
    indices = indices.to(device=device)

    # setup iterations
    bar = tqdm.trange(max_num_iter, desc="Optimization")

    for b in range(k_input.shape[0]):
        # batch processing
        # ToDo: better implementation?
        k = k_input[b].clone().to(device).requires_grad_(True)
        sampling_mask_batch = sampling_mask[b].to(device)
        k_sampled_points = k_input[b].to(device)[sampling_mask_batch]

        # iterations
        for i in bar:
            loss, loss_1, loss_2 = func_optim(
                k=k, indices=indices, s_threshold=s_threshold,
                k_sampled_points=k_sampled_points, sampling_mask=sampling_mask_batch, lam_s=lam_s,
                q=q, matrix_shape=loraks_matrix_shape
            )
            loss.backward()

            # Use the optimal learning_rate to update parameters
            if i % 4 == 0:
                with torch.no_grad():
                    k -= lr[i] * k.grad
                k.grad.zero_()

            bar.postfix = (
                f"loss low rank: {1e3 * loss_1.item():.2f} -- loss data: {1e3 * loss_2.item():.2f} -- "
                f"total_loss: {1e3 * loss.item():.2f}"
            )
        # k is our converged best guess candidate, need to unprep / reshape
        k_out[b] = k.detach().cpu()

    k_out = reshape_k_shape(k_shape=k_out)

    return k_out, k_input


def main():
    # set output path
    path_out = plib.Path("./dev_sim/loraks").absolute()
    path_out.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")
    # torch.cuda.memory._record_memory_history()

    # setup phantom
    num_coils = 8
    num_echoes = 4
    k_load, sampling_mask = create_phantom(nx=256, ny=256, nc=num_coils, ne=num_echoes)
    shape = k_load.shape  # dims [nx, ny, nz, nc, ne, m] - assumed input

    # set LORAKS parameters
    rank = 30
    lam_s = 0.1
    max_num_iter = 400
    # torch.cuda.memory._record_memory_history()
    logging.info(f"Setup LORAKS: Rank - {rank}")
    # use adaptive learning rate
    lr = torch.linspace(5e-3, 1e-4, max_num_iter)
    # lr = torch.full((max_num_iter,), 1e-3)
    # lr = torch.exp(-torch.arange(max_num_iter)/max_num_iter) * 5e-3

    oversampling_factor = 10
    q = rank + oversampling_factor

    # build s_threshold based on rank
    # low rank svd method oversampling factor of 10
    # - note: when using standard SVD we need the full matrix rank (smaller dim)
    s_threshold = torch.ones(q, dtype=torch.float32)
    s_threshold[rank:] = 0
    s_threshold = s_threshold.to(device)

    k_recon_comp_js, k_in_js = comparison_js(
        k_load=k_load, sampling_mask=sampling_mask, lr=lr, device=device,
        max_num_iter=max_num_iter, q=q, s_threshold=s_threshold, lam_s=lam_s
    )
    k_recon_core, k_in_core = core(
        k_load=k_load, sampling_mask=sampling_mask, lr=lr, device=device,
        max_num_iter=max_num_iter, q=q, s_threshold=s_threshold, lam_s=lam_s
    )
    assert torch.allclose(k_in_core, k_in_js)
    # output should be [nx, ny, nz, nc, ne, m]

    # plotting
    logging.info("fft + rsos us")
    img_us = fft(torch.squeeze(k_load).to(device), axes=(0, 1))
    img_us = torch.abs(img_us)
    if num_coils > 1:
        img_us = root_sum_of_squares(img_us, dim_channel=2)
    if num_echoes is not None:
        img_us = img_us[..., 0]

    logging.info("fft + rsos core")
    img = fft(torch.squeeze(k_recon_core).to(device), axes=(0, 1))
    img = torch.abs(img)
    img = torch.abs(img)
    if num_coils > 1:
        img = root_sum_of_squares(img, dim_channel=2)
    if num_echoes is not None:
        img = img[..., 0]

    logging.info("fft + rsos js")
    img_js = fft(torch.squeeze(k_recon_comp_js).to(device), axes=(0, 1))
    img_js = torch.abs(img_js)
    if num_coils > 1:
        img_js = root_sum_of_squares(img_js, dim_channel=2)
    if num_echoes is not None:
        img_js = img_js[..., 0]

    logging.info("plot")
    fig = psub.make_subplots(
        rows=1, cols=3, column_titles=["input", "core", "js"]
    )
    fig.add_trace(
        go.Heatmap(z=img_us.cpu(), colorscale="Gray"), row=1, col=1
    )
    xaxis = fig.data[-1].xaxis
    fig.update_yaxes(visible=False, row=1, col=1, scaleanchor=xaxis, scaleratio=1)
    fig.add_trace(
        go.Heatmap(z=img.cpu(), colorscale="Gray"), row=1, col=2
    )
    xaxis = fig.data[-1].xaxis
    fig.update_yaxes(visible=False, row=1, col=2, scaleanchor=xaxis, scaleratio=1)
    fig.add_trace(
        go.Heatmap(z=img_js.cpu(), colorscale="Gray"), row=1, col=3
    )
    xaxis = fig.data[-1].xaxis
    fig.update_yaxes(visible=False, row=1, col=3, scaleanchor=xaxis, scaleratio=1)
    fig.update_xaxes(visible=False)
    fig_name = path_out.joinpath(f"c_indexing_comp").with_suffix(".html")
    logging.info(f"Write file: {fig_name}")
    fig.write_html(fig_name.as_posix())


# Notes / ToDos / code snippet archive:
# use page implementation - did it wrong but still smoothes the loss evolution :D
# http://proceedings.mlr.press/v139/li21a/li21a.pdf
# use bpcgga from https://doi.org/10.1016/j.neucom.2017.08.037
def conjugate_gd_and_plot_snippets():
    # # get indices for operators - direction is along x and y, set those to 1
    # indices_mapping, batch_reshape = get_all_idx_nd_square_patches_in_nd_shape(
    #     size=loraks_nb_side_length, patch_direction=(1, 1, 0, 0, 0, 0), k_space_shape=shape,
    #     combination_direction=(0, 0, 0, 1, 1, 1)
    # )
    # # returns dims [b (non patch, non combination dims), n_patch, n_combination + neighborhood]
    # # need to reshape the input k-space
    # k_input = torch.reshape(k_input, batch_reshape)
    # sampling_mask = torch.reshape(sampling_mask, batch_reshape)
    #
    # # get LORAKS matrix dimensions
    # n_spatial = indices_mapping.shape[1]
    # n_nb = indices_mapping.shape[2]
    #
    # matrix_rank = min(n_spatial, n_nb)
    #
    # # build s_threshold based on rank
    # s_threshold = torch.ones(matrix_rank, dtype=torch.float32)
    # s_threshold[rank:] = 0
    # s_threshold = s_threshold.to(device)
    #
    # # log losses and plot data
    # losses = []
    #
    # # plot intermediates
    # sl_us_img = fft(k_input, img_to_k=False, axes=(0, 1))
    # sl_us_img = root_sum_of_squares(sl_us_img, dim_channel=-3)
    # plot_k = [k_input[:, :, 0, 0, 0, 0].cpu()]
    # plot_img = [sl_us_img[:, :, 0, 0, 0].cpu()]
    # plot_grads = [torch.zeros_like(plot_k[-1])]
    # plot_names = [0]
    #
    # # setup iterations
    # bar = tqdm.trange(max_num_iter, desc="Optimization")
    #
    #
    # for b in range(k_input.shape[0]):
    #     k = k_input[b].clone().to(device).requires_grad_(True)
    #     index_batch = indices_mapping[b].to(device)
    #     sampling_mask_batch = sampling_mask[b].to(device)
    #     k_input_batch = k_input[b].to(device)[sampling_mask_batch]
    #     prev_gradient = torch.zeros_like(k)
    #
    #     for i in bar:
    #         loss, loss_1, loss_2 = func_optim(
    #             k=k, indices=index_batch, s_threshold=s_threshold,
    #             k_sampled_points=k_input_batch, sampling_mask=sampling_mask_batch, lam_s=lam_s, rank=rank
    #         )
    #         loss.backward()
    #
    #         grad = k.grad  # Gradienten des aktuellen Schritts
    #
    #         if i == 0 or search_direction is None:
    #             # Zu Beginn wird die Suchrichtung als negativer Gradient initialisiert
    #             search_direction = -grad
    #             beta = 0  # Kein Beta-Wert beim ersten Schritt
    #         else:
    #             # Beta berechnen
    #             beta = torch.dot(grad.flatten(), grad.flatten()) / torch.dot(prev_gradient.flatten(),
    #                                                                          prev_gradient.flatten())
    #             # Konjugierte Suchrichtung aktualisieren
    #             search_direction = -grad + beta * search_direction
    #
    #         # Lernrate mit Armijo-Zeilensuche bestimmen
    #         learning_rate = armijo_line_search(
    #             k=k, loss=loss, grad=grad, direction=search_direction, func_optim=func_optim,
    #             index_batch=index_batch, s_threshold=s_threshold, k_input_batch=k_input_batch,
    #             sampling_mask_batch=sampling_mask_batch, lam_s=lam_s, rank=rank
    #         )
    #         # learning_rate = armijo_search(loss_func=func, param=k, direction=search_direction, grad=grad)
    #         # learning_rate = lr[i]
    #         # search_direction = -k.grad
    #
    #         # Use the optimal learning_rate to update parameters
    #         with torch.no_grad():
    #             k += learning_rate * search_direction
    #             grads = torch.abs(k.grad)
    #             conv = torch.linalg.norm(grads - prev_gradient)
    #             prev_gradient = grad.clone()
    #
    #             if i in np.unique(np.logspace(0.1, np.log2(max_num_iter), 10, base=2, endpoint=True).astype(int) - 1):
    #                 # save for plotting on some iterations
    #                 grads = grads[:, :, 0, 0].cpu()
    #
    #                 # some plotting intermediates
    #                 k_recon_loraks = k.clone().detach()
    #                 p_k = k_recon_loraks[:, :, :, 0, 0].clone().detach().cpu()
    #                 img = fft(p_k, axes=(0, 1), img_to_k=False).cpu()
    #
    #                 plot_k.append(p_k[:, :, 0])
    #                 plot_img.append(root_sum_of_squares(img, dim_channel=-1))
    #                 plot_grads.append(grads[:, :, 0])
    #                 plot_names.append(i + 1)
    #
    #
    #         k.grad.zero_()
    #
    #         losses.append(
    #             {"total": loss.item(), "data": loss_2.item(), "low rank": loss_1.item(),
    #              "conv": conv.cpu().item()}
    #         )
    #
    #         bar.postfix = (
    #             f"loss low rank: {1e3 * loss_1.item():.2f} -- loss data: {1e3 * loss_2.item():.2f} -- "
    #             f"total_loss: {1e3 * loss.item():.2f} -- conv : {conv.item()} -- rank: {rank}"
    #         )
    #
    # # aftermath
    #
    # # path_memory_snapshot = path_out.joinpath(f"memory_snapshot").with_suffix(".pickle")
    # # logging.info(f"Write file: {path_memory_snapshot}")
    # # torch.cuda.memory._dump_snapshot(path_memory_snapshot.as_posix())
    #
    # fig = psub.make_subplots(
    #     rows=3, cols=len(plot_k),
    #     column_titles=plot_names
    # )
    # for i, pk in enumerate(plot_k):
    #     fig.add_trace(
    #         go.Heatmap(z=torch.abs(plot_img[i]).numpy(), showscale=False),
    #         row=1, col=1 + i
    #     )
    #     fig.add_trace(
    #         go.Heatmap(z=torch.log(torch.abs(pk)).numpy(), showscale=False),
    #         row=2, col=1 + i
    #     )
    #     fig.add_trace(
    #         go.Heatmap(z=torch.abs(plot_grads[i]).numpy(), showscale=False),
    #         row=3, col=1 + i
    #     )
    # fig.update_xaxes(visible=False)
    # fig.update_yaxes(visible=False)
    # fig_name = path_out.joinpath(f"k_optim").with_suffix(".html")
    # logging.info(f"Write file: {fig_name}")
    # fig.write_html(fig_name.as_posix())
    #
    # losses = pl.DataFrame(losses)
    #
    # fig = go.Figure()
    # fig.add_trace(
    #     go.Scattergl(y=losses["total"], name="total loss", mode="lines")
    # )
    # fig.add_trace(
    #     go.Scattergl(y=losses["data"], name="data", mode="lines")
    # )
    # fig.add_trace(
    #     go.Scattergl(y=losses["low rank"], name="low rank", mode="lines")
    # )
    # fig.add_trace(
    #     go.Scattergl(y=losses["conv"], name="conv", mode="lines")
    # )
    # fig_name = path_out.joinpath(f"losses").with_suffix(".html")
    # logging.info(f"Write file: {fig_name}")
    # fig.write_html(fig_name.as_posix())
    #
    # # recon_k = k.detach()
    # # recon_img = fft(recon_k, img_to_k=False, axes=(0, 1))
    # # recon_img = root_sum_of_squares(recon_img, dim_channel=-2)
    pass


def armijo_search(
        loss_func, param, direction, grad,
        mu_1: float = 0.5, gamma_1: float = 1e-3,
        learning_rate=0.5, gamma: float = 0.5,
        max_iter: int = 20):
    """Performs an Armijo Line Search
    Args:
        loss_func (callable): Function that computes the loss.
        param (torch.Tensor): Parameter being optimized.
        direction (torch.Tensor): search direction of the parameter change.
        learning_rate (float, optional): Starting learning rate. Defaults to 1.0.
        gamma (float, optional): Learning rate reduction factor. Defaults to 0.8.
    Returns:
        float: Optimal learning rate.
    """
    lp = loss_func(param)
    for i in range(max_iter):
        new_param = param + learning_rate * direction
        a = loss_func(new_param)
        b = lp + mu_1 * learning_rate * torch.linalg.norm(direction * grad)
        if a <= b and learning_rate > gamma_1:
            break
        if learning_rate < 1e-4:
            break
        learning_rate *= gamma
    return learning_rate


def armijo_line_search(k, loss, grad, direction, func_optim, index_batch, s_threshold, k_input_batch,
                       sampling_mask_batch, lam_s, rank, alpha=0.5, gamma=0.1):
    """
    Armijo-Zeilensuche, um die optimale Lernrate zu bestimmen.
    Parameters:
        k: aktueller Tensor.
        loss: Loss (vorheriger Wert).
        grad: Gradient.
        direction: Suchrichtung (conjugate gradient).
        func_optim: Optimierungsfunktion.
        alpha: Reduktionsfaktor der Schrittgröße (zwischen 0 und 1).
        gamma: Kontrollparameter (meistens kleines, positives Skalar).
    """
    learning_rate = 1.0  # Start mit anfänglicher Lernrate
    c = gamma

    # Armijo-Bedingung prüfen
    while True:
        # Vorschlag zur nächsten Position
        k_new = k + learning_rate * direction

        # Neue Loss nach Änderung auswerten
        with torch.no_grad():
            new_loss, _, _ = func_optim(
                k=k_new, indices=index_batch, s_threshold=s_threshold,
                sl_us_k=k_input_batch, sampling_mask=sampling_mask_batch, lam_s=lam_s, rank=rank
            )

        # Armijo-Bedingung:
        # new_loss <= loss + c * learning_rate * torch.dot(grad.flatten(), direction.flatten())
        if new_loss <= loss + c * learning_rate * torch.abs(torch.dot(grad.flatten(), direction.flatten())):
            break  # Bedingung erfüllt

        # Schrittweite reduzieren
        learning_rate *= alpha

        # Abbruchbedingung: Sehr kleine Lernrate
        if learning_rate < 1e-8:
            break

    return learning_rate


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    main()
