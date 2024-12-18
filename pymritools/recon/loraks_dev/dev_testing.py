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
from pymritools.recon.loraks_dev.matrix_indexing import get_all_idx_nd_square_patches_in_nd_shape
from pymritools.recon.loraks_dev.operators import s_operator, s_adjoint_operator, c_operator, c_adjoint_operator
from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd, cgd


def func_optim(k, indices, s_threshold, sl_us_k, sampling_mask, lam_s, rank):
    # get operator matrix
    matrix = s_operator(
        k_space=k, indices=indices
    )

    # do svd
    # we can use torch svd, or try the randomized versions
    # u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    u, s, vh = randomized_svd(matrix, sampling_size=rank, oversampling=2*rank, power_projections=2)
    # u, s, vh = subspace_orbit_randomized_svd(matrix, rank=rank)

    # threshold singular values
    # s_r = s * s_threshold

    # reconstruct the low rank approximation
    # u * s @ vh
    matrix_recon_loraks = torch.matmul(u * s.to(u.dtype), vh)

    # __ Low Rank Loss
    # enforce low rank matrix structure by minimizing distance of LORAKS matrix to its truncated SVD
    loss_1 = torch.linalg.norm(matrix - matrix_recon_loraks, ord="fro")

    # __ Data Loss
    # enforce data - consistency by minimizing distance to known sampled points
    loss_2 = torch.linalg.norm(k[sampling_mask] - sl_us_k[sampling_mask])

    return loss_2 + lam_s * loss_1, loss_1, loss_2


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

    logging.info("add misc dimension and input shape")
    # We want to set LORAKS input dimensions to be (nx, ny, nz, nc, ne, m)
    sl_us_k = sl_us_k[:, :, None, :, :, None]
    shape = sl_us_k.shape

    # set LORAKS parameters
    loraks_nb_side_length = 5
    rank = 50
    lam_s = 0.05
    max_num_iter = 200
    device = torch.device("cuda")
    # torch.cuda.memory._record_memory_history()

    logging.info(f"Setup LORAKS: Rank - {rank}")

    # use adaptive learning rate
    # lr = np.linspace(0.008, 0.0005, max_num_iter)
    lr = torch.exp(-torch.arange(max_num_iter)/max_num_iter) * 5e-3

    # __ One Time Calculations __
    # get dimensions
    n_read, n_phase, n_slice, n_channels, n_echoes, n_misc = shape

    # combine dimensions in terms of LORAKS low rank idea:
    # we want patches across spatial dimensions to form neighborhoods.
    # These neighborhoods are duplicated across spatial redundant dimensions (channels, echoes, misc) and combined.
    # This means we concatenate the neighborhoods in these dimensions (neighborhood size = nb).
    # There are several ways to go about this indexing wise. For now we combine to a shape prior to computing indexing.
    # Target shape: batch dim - spatial dims (patches) - combined dims (l): [b, nxyz, ncem * nb] = [b, np, l * nb]
    # If the Target shape doesnt fit GPU memory whole, we can use the independent batch dimension
    # to break down the size for computations.
    # If a singular [1, nxyz, ncem] doesnt fit GPU memory due to size, we need to adress the combination method
    # of c-e-m or reduce neighborhood size. Both need some addressing carefully and optimization evaluation.
    # For now we have virtual phantom data and stick to full combination of sufficiently low data sizes.
    # Other speed optimization will depend on the matrix size (eg. SVD) and hence be dependent on the combination dim.

    # __ Dim combination
    # This section still needs to be filled with convenient methods for combination.
    # For now we decide that one spatial dimension (conveniently z) is batched and
    # combine all spatial and all redundancy dimensions.
    # There might be various situations were this is not wanted (i.e. 3D patches, other imaging gradient directions) or
    # not possible (matrix dimensions and GPU memory trouble).
    # k_input = sl_us_k.movedim(2, 0)
    # k_input = k_input.view(n_slice, n_read, n_phase, -1)
    #
    # # calculate matrix dimensions
    # # long side is all valid spatial points that support building a full patch within the shape
    # m = (n_read - radius + 1) * (n_phase - radius + 1)
    # # short side depends on the combination of the channels, coils, neighborhood dimensions and
    # # LORAKS matrix (S = factor 2)
    # loraks_method_factor = 1
    # n = nb * loraks_method_factor


    # get indices for operators - direction is along x and y, set those to 1
    indices_mapping, batch_reshape = get_all_idx_nd_square_patches_in_nd_shape(
        size=loraks_nb_side_length, patch_direction=(1, 1, 0, 0, 0, 0), k_space_shape=sl_us_k.shape,
        combination_direction=(0, 0, 0, 1, 1, 1)
    )
    # returns dims [b (non patch, non combination dims), n_patch, n_combination + neighborhood]
    # need to reshape the input k-space
    k_input = torch.reshape(sl_us_k, batch_reshape)

    # get sampling mask in input shape
    sampling_mask = (torch.abs(k_input) > 1e-9)

    # embed image in random image to ensure svd gradient stability
    k_init = torch.randn_like(k_input)
    k_init *= 1e-2 * torch.max(torch.abs(k_input)) / torch.max(torch.abs(k_init))
    k_init[sampling_mask] = k_input[sampling_mask]

    logging.info("__ Testing algorithm")
    n_spatial = indices_mapping.shape[1]
    n_nb = indices_mapping.shape[2]

    matrix_rank = min(n_spatial, n_nb)

    # build s_threshold based on rank
    s_threshold = torch.ones(matrix_rank, dtype=torch.float32)
    s_threshold[rank:] = 0
    s_threshold = s_threshold.to(device)

    # log losses
    losses = []

    # plot intermediates
    sl_us_img = fft(sl_us_k, img_to_k=False, axes=(0, 1))
    sl_us_img = root_sum_of_squares(sl_us_img, dim_channel=-3)
    plot_k = [sl_us_k[:, :, 0, 0, 0, 0].cpu()]
    plot_img = [sl_us_img[:, :, 0, 0, 0].cpu()]
    plot_grads = [torch.zeros_like(plot_k[-1])]
    plot_names = [0]

    bar = tqdm.trange(max_num_iter, desc="Optimization")
    # use page implementation - did it wrong but still smoothes the loss evolution :D
    # http://proceedings.mlr.press/v139/li21a/li21a.pdf
    p_t = 0.4

    for b in range(k_input.shape[0]):
        k = k_init[b].clone().to(device).requires_grad_(True)
        index_batch = indices_mapping[b].to(device)
        sampling_mask_batch = sampling_mask[b].to(device)
        k_input_batch = k_input[b].to(device)
        grad_last = torch.zeros_like(k)

        for i in bar:
            loss, loss_1, loss_2 = func_optim(
                k=k, indices=index_batch, s_threshold=s_threshold,
                sl_us_k=k_input_batch, sampling_mask=sampling_mask_batch, lam_s=lam_s, rank=rank
            )
            loss.backward()
            # use bpcgga from https://doi.org/10.1016/j.neucom.2017.08.037
            # grad = k.grad.clone()
            # if i == 0:
            #     search_direction = -grad
            # else:
                # # get beta
                # cos_theta = torch.dot(search_direction.view(-1), grad.view(-1)) / (torch.norm(search_direction) * torch.norm(grad))
                # norms = torch.linalg.norm(grad) / torch.linalg.norm(search_direction)
                # beta_u = norms / (1 + 1e-9 + cos_theta)
                # beta_l = - norms / (1 + 1e-9 - cos_theta)
                # # sample beta from range
                # beta = (beta_u - beta_l) * torch.rand_like(beta_l) + beta_l
                # # compute new direction
                # search_direction = - grad + beta * search_direction

            # learning_rate = armijo_search(loss_func=func, param=k, direction=search_direction, grad=grad)
            learning_rate = lr[i]
            search_direction = -k.grad

            # Use the optimal learning_rate to update parameters
            with torch.no_grad():
                k += learning_rate * search_direction
                # grad_update = k.grad
                # page_mask = torch.rand(grad_update.shape) > p_t
                # grad = grad_update - grad_last
                # grad[page_mask] = grad_update[page_mask]
                # k -= lr[i] * grad
                # grad_last = grad_update

                grads = torch.abs(k.grad)
                conv = torch.linalg.norm(grads - grad_last).cpu()
                grad_last = grads.clone()
                grads = grads[:, :, 0, 0].cpu()

            k.grad.zero_()

            losses.append(
                {"total": loss.item(), "data": loss_2.item(), "low rank": loss_1.item(),
                 "conv": conv.item()}
            )

            bar.postfix = (
                f"loss low rank: {1e3 * loss_1.item():.2f} -- loss data: {1e3 * loss_2.item():.2f} -- "
                f"total_loss: {1e3 * loss.item():.2f} -- conv : {conv.item()} -- rank: {rank}"
            )

            if i in np.unique(np.logspace(0.1, np.log2(max_num_iter), 10, base=2, endpoint=True).astype(int)-1):
                # some plotting intermediates
                k_recon_loraks = k.clone().detach()
                p_k = k_recon_loraks[:, :, :, 0, 0].clone().detach().cpu()
                img = fft(p_k, axes=(0, 1), img_to_k=False).cpu()

                plot_k.append(p_k[:, :, 0])
                plot_img.append(root_sum_of_squares(img, dim_channel=-1))
                plot_grads.append(grads[:, :, 0])
                plot_names.append(i+1)

    # path_memory_snapshot = path_out.joinpath(f"memory_snapshot").with_suffix(".pickle")
    # logging.info(f"Write file: {path_memory_snapshot}")
    # torch.cuda.memory._dump_snapshot(path_memory_snapshot.as_posix())

    fig = psub.make_subplots(
        rows=3, cols=len(plot_k),
        column_titles=plot_names
    )
    for i, pk in enumerate(plot_k):
        fig.add_trace(
            go.Heatmap(z=torch.abs(plot_img[i]).numpy(), showscale=False),
            row=1, col=1 + i
        )
        fig.add_trace(
            go.Heatmap(z=torch.log(torch.abs(pk)).numpy(), showscale=False),
            row=2, col=1 + i
        )
        fig.add_trace(
            go.Heatmap(z=torch.abs(plot_grads[i]).numpy(), showscale=False),
            row=3, col=1 + i
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
