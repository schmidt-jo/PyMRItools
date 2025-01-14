"""
Want to create a test case for AC-LORAKS, create current benchmark dataset and implement
the changes in matrix indexing and operators from LORAKS into AC-LORAKS that are transferable.
Then test against the dataset to ensure equality.
___
Jochen Schmidt
14.01.2025
"""
import os

import torch
import plotly.subplots as psub
import plotly.graph_objects as go

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft, root_sum_of_squares
from pymritools.recon.loraks.algorithms import ac_loraks

from tests.utils import get_test_result_output_dir


def create_phantom(nx: int, ny: int, nc: int, ne: int):
    sl_fs = SheppLogan().get_2D_k_space(
        shape=(nx, ny), num_coils=nc, num_echoes=ne
    )
    sl = SheppLogan().get_sub_sampled_k_space(
        shape=(nx, ny), acceleration=3, ac_lines=30, mode='skip', num_coils=nc, num_echoes=ne
    )
    fs_img = fft(sl_fs, axes=(0, 1))
    fs_img = root_sum_of_squares(fs_img, dim_channel=-2)

    us_img = fft(sl, axes=(0, 1))
    us_img = root_sum_of_squares(us_img, dim_channel=-2)

    return sl, sl_fs, us_img, fs_img


def test_ac_loraks():
    # set output directory
    output_dir = get_test_result_output_dir(test_ac_loraks)
    # get phantom
    nx, ny, nc, ne = (256, 256, 32, 4)
    sl_us, sl_fs, us_img, fs_img = create_phantom(nx=nx, ny=ny, nc=nc, ne=ne)
    # set loraks params
    radius = 3
    s_rank = 100
    s_lambda = 0.05
    max_num_iter = 20

    sampling_mask = torch.abs(sl_us) > 1e-9
    torch.cuda.memory._record_memory_history()

    recon_k = ac_loraks(
        k_space_x_y_z_ch_t=sl_us[:, :, None], sampling_mask_x_y_t=sampling_mask[:, :, 0],
        radius=radius, rank_c=100, lambda_c=0.0, rank_s=s_rank, lambda_s=s_lambda,
        batch_size_channels=32, max_num_iter=max_num_iter,
        visualize=True, path_visuals=os.path.join(output_dir, "figures"),
        device=torch.device("cuda:0")
    )
    torch.cuda.memory._dump_snapshot(os.path.join(output_dir, "memory_snapshot.pickle"))

    recon_img = fft(recon_k, axes=(0, 1))
    recon_img = torch.squeeze(root_sum_of_squares(recon_img, dim_channel=-2))

    # write to file
    filename = os.path.join(output_dir, "recon_k.pt")
    torch.save(recon_k, filename)
    filename = os.path.join(output_dir, "recon_img.pt")
    torch.save(recon_img, filename)
    filename = os.path.join(output_dir, "sl_fs_img.pt")
    torch.save(fs_img, filename)
    filename = os.path.join(output_dir, "sl_us_img.pt")
    torch.save(us_img, filename)

    # visualize
    fig = psub.make_subplots(
        rows=1, cols=3,
        column_titles=["fs", "us", "recon"],
        shared_yaxes=True,
    )
    for i, d in enumerate([fs_img, us_img, recon_img]):
        fig.add_trace(
            go.Heatmap(z=d[:, :, 0], transpose=False, colorscale="Gray"),
            row=1, col=1+i
        )
        xaxis=fig.data[-1].xaxis
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor=xaxis)

    filename = os.path.join(output_dir, "comp_ac_loraks.html")
    fig.write_html(filename)
