"""
Want to create a test case for AC-LORAKS, create current benchmark dataset and implement
the changes in matrix indexing and operators from LORAKS into AC-LORAKS that are transferable.
Then test against the dataset to ensure equality.
___
Jochen Schmidt
14.01.2025
"""
import os
from scipy.io import savemat

import torch
import plotly.subplots as psub
import plotly.graph_objects as go

from pymritools.utils import Phantom
from pymritools.utils import fft_to_img, root_sum_of_squares
from pymritools.recon.loraks.algorithms import ac_loraks

from tests.utils import get_test_result_output_dir


def create_phantom(nx: int, ny: int, nc: int, ne: int):
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)
    sl_fs = phantom.get_2d_k_space()
    sl = phantom.sub_sample_ac_random_lines(acceleration=4, ac_lines=42)

    fs_img = fft(sl_fs, axes=(0, 1))
    if nc > 1:
        fs_img = root_sum_of_squares(fs_img, dim_channel=-2)
    else:
        fs_img = torch.abs(fs_img)

    us_img = fft(sl, axes=(0, 1))
    if nc > 1:
        us_img = root_sum_of_squares(us_img, dim_channel=-2)
    else:
        us_img = torch.abs(us_img)

    return sl, sl_fs, us_img, fs_img


def test_ac_loraks():
    # set output directory
    output_dir = get_test_result_output_dir(test_ac_loraks)
    # get phantom
    nx, ny, nc, ne = (256, 256, 1, 20)
    sl_us, sl_fs, us_img, fs_img = create_phantom(nx=nx, ny=ny, nc=nc, ne=ne)
    # set loraks params
    radius = 3
    s_rank = 200
    max_num_iter = 20

    sampling_mask = torch.abs(sl_us) > 1e-9
    # torch.cuda.memory._record_memory_history()

    # recon_k = ac_loraks(
    #     k_space_x_y_z_ch_t=sl_us[:, :, None, None], sampling_mask_x_y_t=sampling_mask[:, :, :],
    #     radius=radius, rank_c=100, lambda_c=None, rank_s=s_rank, lambda_s=s_lambda,
    #     batch_size_channels=3, max_num_iter=max_num_iter,
    #     visualize=True, path_visuals=os.path.join(output_dir, "figures"),
    #     device=torch.device("cuda:0")
    # )
    recon_k = ac_lorals_exact_data_consistency(
        k_space_x_y_z_ch_t=sl_us[:, :, None, None], sampling_mask_x_y_t=sampling_mask[:, :, :],
        radius=radius, rank_s=s_rank,
        batch_size_channels=1, max_num_iter=max_num_iter,
        visualize=True, path_visuals=os.path.join(output_dir, "figures"),
        device=torch.device("cuda:0")
    )
    # torch.cuda.memory._dump_snapshot(os.path.join(output_dir, "memory_snapshot.pickle"))

    recon_img = fft_to_img(recon_k, axes=(0, 1))
    recon_img = torch.squeeze(root_sum_of_squares(recon_img, dim_channel=-2))

    # write to file
    # filename = os.path.join(output_dir, "recon_k.pt")
    # torch.save(recon_k, filename)
    # filename = os.path.join(output_dir, "recon_img.pt")
    # torch.save(recon_img, filename)
    # filename = os.path.join(output_dir, "sl_fs_img.pt")
    # torch.save(fs_img, filename)
    # filename = os.path.join(output_dir, "sl_us_img.pt")
    # torch.save(us_img, filename)

    nifti_save(recon_img, img_aff=torch.eye(4), path_to_dir=os.path.abspath(output_dir), file_name="recon_img.nii")
    savemat(os.path.join(output_dir, "sl_us.mat"), {"sl_us": sl_us.cpu().numpy()})

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
