import os.path

import torch
import plotly.subplots as psub
import plotly.graph_objects as go
from numpy.ma.core import shape

from pymritools.utils import fft_to_img, root_sum_of_squares, Phantom
from pymritools.recon.loraks.algorithms import ac_loraks
from tests.utils import get_test_result_output_dir


def set_axis(fig, row, col):
    xaxis=fig.data[-1].xaxis
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, scaleanchor=xaxis, row=row, col=col)


def compare_sampling_patterns():
    # get phantom with medium high dimensionality
    nx, ny, nc, ne = (256, 256, 8, 4)
    phantom = Phantom.get_shepp_logan(shape=(nx, ny), num_coils=nc, num_echoes=ne)

    sl_gt = phantom.get_2d_k_space()
    img_gt = torch.abs(fft_to_img(sl_gt, axes=(0, 1)))
    # img_gt = root_sum_of_squares(img_gt, dim_channel=-2)

    modes = ["grappa", "skip", "random", "weighted"]

    fig = psub.make_subplots(
        rows=5, cols=len(modes)+1,
        column_titles=["gt"] + modes + ["diff"],
        row_titles=["sampling", "us img", "recon img", "error % diff", "rSoS"],
        horizontal_spacing=0.01, vertical_spacing=0.01,
    )
    fig.add_trace(
        go.Heatmap(z=torch.ones_like(img_gt[:, :, 0, 0]), zmin=0, zmax=1, showscale=False, colorscale="Magma"),
        row=1, col=1
    )
    zmax_img = 0.5
    set_axis(fig, 1, 1)
    for r in range(2):
        fig.add_trace(
            go.Heatmap(z=img_gt[:, :, 0, 0], zmin=0, zmax=zmax_img,
                       showscale=False, colorscale="gray"),
            row=r+2, col=1
        )
        set_axis(fig, r+2, 1)

    fig.add_trace(
        go.Heatmap(z=torch.zeros_like(img_gt[:, :, 0, 0]), zmin=0, zmax=1, showscale=False, colorscale="Magma"),
        row=4, col=1
    )
    zmax_img = 0.4
    set_axis(fig, 4, 1)
    fig.add_trace(
        go.Heatmap(z=root_sum_of_squares(img_gt[:, :, :, 0], dim_channel=-1), zmin=0, zmax=zmax_img, showscale=False, colorscale="gray"),
        row=5, col=1
    )
    zmax_img = 0.4
    set_axis(fig, 5, 1)

    for mi, m in enumerate(modes):
        # get undersampled phantom
        match m:
            case "grappa":
                sl = phantom.sub_sample_ac_grappa(acceleration=4, ac_lines=30)
            case "skip":
                sl = phantom.sub_sample_ac_skip(acceleration=4, ac_lines=30)
            case "random":
                sl = phantom.sub_sample_ac_random(acceleration=4, ac_lines=30)
            case "weighted":
                sl = phantom.sub_sample_ac_weighted(acceleration=4, ac_lines=30)

        mask = (torch.abs(sl) > 1e-9).to(torch.int)

        fig.add_trace(
            go.Heatmap(z=mask[:, :, 0, 0], zmin=0, zmax=1, showscale=False, colorscale="Magma"),
            row=1, col=mi+2
        )
        set_axis(fig, 1, mi+2)

        img = torch.abs(fft_to_img(sl, axes=(0, 1)))
        # img = root_sum_of_squares(img, dim_channel=-2)
        fig.add_trace(
            go.Heatmap(z=img[:, :, 0, 0], showscale=False, colorscale="gray", zmin=0, zmax=zmax_img),
            row=2, col=mi+2
        )
        set_axis(fig, 2, mi+2)

        recon = torch.squeeze(
            ac_loraks(
                k_space_x_y_z_ch_t=sl[:, :, None], sampling_mask_x_y_t=mask[:, :, 0],
                radius=3, rank_c=50, lambda_c=0.0, rank_s=150, lambda_s=0.05, batch_size_channels=100,
                max_num_iter=15, visualize=False, path_visuals="", device=torch.device("cuda")
            )
        )
        img_recon = torch.abs(fft_to_img(recon, axes=(0, 1)))
        # img_recon = root_sum_of_squares(img_recon, dim_channel=-2)
        fig.add_trace(
            go.Heatmap(z=img_recon[:, :, 0, 0], showscale=False, colorscale="gray", zmin=0, zmax=zmax_img),
            row=3, col=mi+2
        )
        set_axis(fig, 3, mi+2)
        diff = 100 * torch.nan_to_num((img_recon - img_gt) / img_gt, nan=0.0, posinf=0.0, neginf=0.0)
        fig.add_trace(
            go.Heatmap(z=diff[:, :, 0, 0], showscale=True, colorscale="Turbo", colorbar=dict(title="% Error"), zmin=-10, zmax=10),
            row=4, col=mi+2
        )
        set_axis(fig, 4, mi+2)

        fig.add_trace(
            go.Heatmap(
                z=root_sum_of_squares(img_recon, dim_channel=-2)[:, :, 0],
                showscale=False, colorscale="gray", zmax=zmax_img),
            row=5, col=mi+2
        )
        set_axis(fig, 5, mi+2)

    fig.update_layout(
        width=1500, height=1500,
    )
    output_dir = get_test_result_output_dir(compare_sampling_patterns)
    file_name = os.path.join(output_dir, "sampling_pattern_comp.html")
    fig.write_html(file_name)


if __name__ == '__main__':
    compare_sampling_patterns()