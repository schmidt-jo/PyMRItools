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
from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd


def func_optim(k, indices, s_threshold, shape, count_matrix, sl_us_k, sampling_mask, lam_s, rank):
    # get operator matrix
    matrix = s_operator(
        k_space=k, indices=indices
    )

    # do svd
    # we can use torch svd, or try the randomized version, see above
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    # u, s, vh = randomized_svd(matrix, sampling_size=rank, oversampling=2*rank)
    # u, s, vh = subspace_orbit_randomized_svd(matrix, rank=rank)

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
    k_recon_loraks /= count_matrix
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

    logging.info("add misc dimension and input shape")
    # We want to set LORAKS input dimensions to be (nx, ny, nz, nc, ne, m)
    sl_us_k = sl_us_k[:, :, None, :, :, None]
    shape = sl_us_k.shape

    # set LORAKS parameters
    radius = 5
    rank = 30
    lam_s = 0.05
    max_num_iter = 100
    device = torch.device("cpu")
    logging.info(f"Setup LORAKS: Rank - {rank}")

    # use adaptive learning rate
    lr = np.linspace(0.05, 0.001, max_num_iter)

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
    k_input = sl_us_k.movedim(2, 0)
    # k_input = k_input.view(n_slice, n_read, n_phase, -1)
    #
    # # calculate matrix dimensions
    # # long side is all valid spatial points that support building a full patch within the shape
    # m = (n_read - radius + 1) * (n_phase - radius + 1)
    # # short side depends on the combination of the channels, coils, neighborhood dimensions and
    # # LORAKS matrix (S = factor 2)
    # loraks_method_factor = 1
    # n = nb * loraks_method_factor

    # get sampling mask in input shape
    sampling_mask = (torch.abs(k_input) > 1e-9)

    # get indices for operators - direction is along x and y, set those to 1
    indices = get_all_idx_nd_square_patches_in_nd_shape(
        size=radius, patch_direction=(0, 1, 1, 0, 0, 0), k_space_shape=k_input.shape,
        combination_direction=(0, 0, 0, 1, 1, 1)
    )

    # now assume we picked a batch
    index_batch = indices[0]
    k_input_batch = k_input[0]

    # get count matrix from indices, ensure nonzero (if using non rectangular patches)
    count_matrix = torch.bincount(index_batch.view(-1))
    count_matrix[count_matrix == 0] = 1

    # test c mapping
    # assume we picked a batch
    c_matrix = c_operator(k_space=k_input_batch, indices=index_batch)
    # adjoint
    k_recon_c = c_adjoint_operator(c_matrix=c_matrix, indices=index_batch, k_space_dims=k_input_batch.shape)
    k_recon_c /= count_matrix.view(k_input_batch.shape)
    print(torch.allclose(k_recon_c, k_input_batch))

    # test s mapping
    s_matrix = s_operator(k_space=k_input_batch, indices=index_batch)

    # Adjoint
    k_recon_s = s_adjoint_operator(s_matrix=s_matrix, indices=index_batch, k_space_dims=k_input_batch.shape)
    # normalize
    count_matrix_s = 2 * count_matrix.view(k_input_batch.shape)
    k_recon_s /= count_matrix_s

    # test
    print(torch.allclose(k_recon_s, k_input_batch))

    fig = psub.make_subplots(
        rows=3, cols=3
    )
    count_matrix = count_matrix.view(k_input_batch.shape)
    counts = [torch.ones_like(count_matrix[:,:,0,0,0]), count_matrix[:,:,0,0,0], count_matrix_s[:,:,0,0,0]]
    for i, d in enumerate([k_input_batch[:,:,0,0,0], k_recon_c[:,:,0,0,0], k_recon_s[:,:,0,0,0]]):
        fig.add_trace(
            go.Heatmap(z=torch.log(torch.abs(d)).numpy(), showscale=False),
            row=1, col=i+1
        )
        fig.add_trace(
            go.Heatmap(z=(torch.log(torch.abs(d)) - torch.log(torch.abs(k_input_batch[:,:,0,0,0]))).numpy(), showscale=False),
            row=3, col=i+1
        )
        fig.add_trace(
            go.Heatmap(z=counts[i].numpy(), showscale=False),
            row=2, col=i+1
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig_name = path_out.joinpath(f"mapping_tests").with_suffix(".html")
    logging.info(f"Write file: {fig_name}")
    fig.write_html(fig_name.as_posix())

    logging.info("__ Testing algorithm")
    # algorithm for s
    matrix_rank = torch.min(torch.tensor(s_matrix.shape)).item()
    # embed image in random image to ensure svd gradient stability
    k_init = torch.randn_like(k_input)
    k_init *= 1e-3 * torch.max(torch.abs(k_input)) / torch.max(torch.abs(k_init))
    k_init[sampling_mask] = k_input[sampling_mask]
    # want to multiply search candidates by sampling mask to compute data consistency term, cast to numbers
    sampling_mask = sampling_mask.to(device=device)
    # save us data for consistency term, send to device
    k_input = k_input.to(device)

    # build s_threshold based on rank
    s_threshold = torch.ones(matrix_rank, dtype=torch.float32)
    s_threshold[rank:] = 0
    s_threshold = s_threshold.to(device)

    # log losses
    losses = []
    # plot intermediates
    sl_us_img = fft(sl_us_k, img_to_k=False, axes=(0, 1))
    sl_us_img = root_sum_of_squares(sl_us_img, dim_channel=-3)
    plot_k = [sl_us_k[:, :, 0, 0].cpu()]
    plot_img = [sl_us_img[:, :, 0].cpu()]
    plot_grads = [torch.zeros_like(plot_k[-1])]

    bar = tqdm.trange(max_num_iter, desc="Optimization")
    for b in range(k_input.shape[0]):
        k = k_init[b].clone().to(device).requires_grad_(True)
        index_batch = indices[b].to(device)
        sampling_mask_batch = sampling_mask[b].to(device)
        for i in bar:
            loss, loss_1, loss_2 = func_optim(
                k=k, indices=index_batch, s_threshold=s_threshold, shape=k.shape, count_matrix=count_matrix,
                sl_us_k=k_input[b], sampling_mask=sampling_mask_batch, lam_s=lam_s, rank=rank
            )
            loss.backward()

            with torch.no_grad():
                k -= lr[i] * k.grad

                grads = torch.abs(k.grad)
                conv = torch.linalg.norm(grads).cpu()
                grads = grads[:, :, 0, 0].cpu()

            k.grad.zero_()

            # optim.step()
            # optim.zero_grad()
            losses.append(
                {"total": loss.item(), "data": loss_2.item(), "low rank": loss_1.item(),
                 "conv": conv.item()}
            )

            bar.postfix = (
                f"loss low rank: {1e3*loss_1.item():.2f} -- loss data: {1e3*loss_2.item():.2f} -- "
                f"total_loss: {1e3*loss.item():.2f} -- conv : {conv.item()} -- rank: {rank}"
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
