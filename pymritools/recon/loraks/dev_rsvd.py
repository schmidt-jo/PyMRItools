import logging
import pathlib as plib

import torch
import numpy as np

from pymritools.utils.phantom import SheppLogan
from pymritools.utils.funtions import gaussian_2d_kernel
from pymritools.utils import fft, root_sum_of_squares
import plotly.graph_objects as go
import plotly.subplots as psub
from pymritools.utils import get_idx_2d_circular_neighborhood_patches_in_shape
from pymritools.recon.loraks.operators import s_operator, s_adjoint_operator
from pymritools.utils.algorithms import randomized_svd, subspace_orbit_randomized_svd


def main():
    path_out = plib.Path("./dev_sim/loraks").absolute()
    path_out.mkdir(exist_ok=True, parents=True)

    shape = (256, 256)
    num_coils = 8
    num_echoes = 6
    logging.info("get SheppLogan phantom")
    logging.info("add virtual coils")
    sl_img = SheppLogan().get_2D_image(shape=shape, num_coils=num_coils)

    logging.info("add decay echoes")
    # create decay echo images
    sl_img = sl_img[:, :, :, None] * torch.exp(-20 * torch.arange(1, 1 + num_echoes) * 0.009)[None, None, None, :]

    # back to k-space
    sl_k_cs = fft(sl_img, img_to_k=True, axes=(0, 1))
    shape = sl_k_cs.shape
    # get indices
    indices = get_idx_2d_circular_neighborhood_patches_in_shape(shape_2d=shape, nb_radius=3)
    # build s_matrix from k-space
    s_matrix = s_operator(k_space_x_y_ch_t=sl_k_cs, indices=indices)

    # set rank
    rank = 30
    logging.info("randomized svd decomposition")
    # using rank as sampling size -> compression right to the truncated rank dimension
    u_rsvd, s_rsvd, vh_rsvd = randomized_svd(matrix=s_matrix, sampling_size=rank, oversampling=int(rank / 2))

    logging.info("subspace orbit randomized svd decomposition")
    # using rank as sampling size -> compression right to the truncated rank dimension
    u_sorsvd, s_sorsvd, vh_sorsvd = subspace_orbit_randomized_svd(matrix=s_matrix, rank=rank)

    logging.info("svd decomposition")
    u_svd, s_svd, vh_svd = torch.linalg.svd(s_matrix, full_matrices=False)
    # use low rank approx. truncation
    u_svd_lr = u_svd[:, :rank]
    s_svd_lr = s_svd[:rank]
    vh_svd_lr = vh_svd[:rank]

    logging.info("Reconstruct LR")
    recon_lr_matrix = torch.matmul(torch.matmul(u_svd_lr, torch.diag_embed(s_svd_lr).to(u_svd_lr.dtype)), vh_svd_lr)
    recon_lr_k = s_adjoint_operator(s_matrix=recon_lr_matrix, indices=indices, k_space_dims=shape)
    logging.info("Reconstruct rSVD")
    recon_rsvd_matrix = torch.matmul(torch.matmul(u_rsvd, torch.diag_embed(s_rsvd).to(u_rsvd.dtype)), vh_rsvd)
    recon_rsvd_k = s_adjoint_operator(s_matrix=recon_rsvd_matrix, indices=indices, k_space_dims=shape)
    logging.info("Reconstruct sorSVD")
    recon_sorsvd_matrix = torch.matmul(torch.matmul(u_sorsvd, torch.diag_embed(s_sorsvd).to(u_sorsvd.dtype)), vh_sorsvd)
    recon_sorsvd_k = s_adjoint_operator(s_matrix=recon_sorsvd_matrix, indices=indices, k_space_dims=shape)

    logging.info("FFT")
    recon_lr_img = fft(recon_lr_k, img_to_k=False, axes=(0, 1))
    recon_rsvd_img = fft(recon_rsvd_k, img_to_k=False, axes=(0, 1))
    recon_sorsvd_img = fft(recon_sorsvd_k, img_to_k=False, axes=(0, 1))

    logging.info("rSoS")
    recon_lr_rsos_img = root_sum_of_squares(input_data=recon_lr_img, dim_channel=-2)
    recon_rsvd_rsos_img = root_sum_of_squares(input_data=recon_rsvd_img, dim_channel=-2)
    recon_sorsvd_rsos_img = root_sum_of_squares(input_data=recon_sorsvd_img, dim_channel=-2)
    rsos_img = root_sum_of_squares(input_data=sl_img, dim_channel=-2)

    diff_rsvd = torch.nan_to_num(
        (recon_rsvd_rsos_img - recon_lr_rsos_img) / recon_lr_rsos_img,
        posinf=0.0, neginf=0.0, nan=0.0
    ) * 100
    diff_sorsvd = torch.nan_to_num(
        (recon_sorsvd_rsos_img - recon_lr_rsos_img) / recon_lr_rsos_img,
        posinf=0.0, neginf=0.0, nan=0.0
    ) * 100

    logging.info("Plot")
    fig = psub.make_subplots(rows=2, cols=3, column_titles=["svd", "rsvd", "sorsvd"])
    data = [recon_lr_rsos_img, recon_sorsvd_rsos_img, recon_sorsvd_rsos_img]
    diffs = [rsos_img, diff_rsvd, diff_sorsvd]

    for idx_d, d in enumerate(data):
        showscale = True if idx_d == 1 else False
        fig.add_trace(
            go.Heatmap(z=torch.abs(d[:, :, 0]), showscale=False),
            row=1, col=1+idx_d
        )
        if idx_d == 0:
            zmin, zmax = (np.min(diffs[idx_d][:, :, 0].numpy()), np.max(diffs[idx_d][:, :, 0].numpy()))
        else:
            zmin, zmax = (-20, 20)
        fig.add_trace(
            go.Heatmap(z=diffs[idx_d][:, :, 0], zmin=zmin, zmax=zmax, showscale=showscale),
            row=2, col=1+idx_d
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig_name = path_out.joinpath("recon_trunc_vs_rsvd").with_suffix(".html")
    fig.write_html(fig_name.as_posix())


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    main()
