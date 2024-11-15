import logging
import pathlib as plib

import torch

from pymritools.utils.phantom import SheppLogan
from pymritools.utils.funtions import gaussian_2d_kernel
from pymritools.utils import fft, root_sum_of_squares
import plotly.graph_objects as go
import plotly.subplots as psub
from pymritools.utils import get_idx_2d_circular_neighborhood_patches_in_shape
from pymritools.recon.loraks.operators import s_operator, s_adjoint_operator
from pymritools.utils.algorithms import randomized_svd


def main():
    path_out = plib.Path("./dev_sim/loraks").absolute()
    path_out.mkdir(exist_ok=True, parents=True)

    shape = (256, 256)
    logging.info("get SheppLogan phantom")
    sl_img = SheppLogan().get_2D_image(shape=shape)

    logging.info("Build virtual coils and decay echoes")
    coil_sens = torch.zeros((*shape, 4))
    centers = [[80, 80], [80, 200], [200, 80], [200, 200]]
    for i in range(4):
        coil_sens[:, :, i] = gaussian_2d_kernel(size_x=shape[0], size_y=shape[1], center_x=centers[i][0],
                                                center_y=centers[i][1], sigma=(80, 100))
    # create coil images (multiplication in image space, convolution in k-space)
    sl_img = sl_img[:, :, None] * coil_sens
    # create decay echo images
    sl_img = sl_img[:, :, :, None] * torch.exp(-20 * torch.arange(0.009, 0.08, 0.009))[None, None, None, :]
    # back to k-space
    sl_k_cs = fft(sl_img, img_to_k=True, axes=(0, 1))
    shape = sl_k_cs.shape
    # get indices
    indices = get_idx_2d_circular_neighborhood_patches_in_shape(shape_2d=shape, nb_radius=3)
    # build s_matrix from k-space
    s_matrix = s_operator(k_space_x_y_ch_t=sl_k_cs, indices=indices)

    # set rank
    rank = 400
    logging.info("randomized svd decomposition")
    # using rank as sampling size -> compression right to the truncated rank dimension
    u_rsvd, s_rsvd, vh_rsvd = randomized_svd(matrix=s_matrix, sampling_size=rank)

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

    logging.info("FFT")
    recon_lr_img = fft(recon_lr_k, img_to_k=False, axes=(0, 1))
    recon_rsvd_img = fft(recon_rsvd_k, img_to_k=False, axes=(0, 1))

    logging.info("rSoS")
    recon_lr_rsos_img = root_sum_of_squares(input_data=recon_lr_img, dim_channel=-2)
    recon_rsvd_rsos_img = root_sum_of_squares(input_data=recon_rsvd_img, dim_channel=-2)
    diff = torch.nan_to_num(
        (recon_rsvd_rsos_img - recon_lr_rsos_img) / recon_rsvd_rsos_img,
        posinf=0.0, neginf=0.0, nan=0.0
    )

    logging.info("Plot")
    fig = psub.make_subplots(rows=1, cols=3)
    fig.add_trace(
        go.Heatmap(z=torch.abs(recon_lr_rsos_img[:, :, 0])),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=torch.abs(recon_rsvd_rsos_img[:, :, 0])),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=diff[:, :, 0]),
        row=1, col=3
    )
    fig_name = path_out.joinpath("recon_trunc_vs_rsvd").with_suffix(".html")
    fig.write_html(fig_name.as_posix())


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    main()
