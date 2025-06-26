import os

import torch
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.utils import Phantom, fft_to_img, root_sum_of_squares
from .utils import get_test_result_output_dir, ResultMode


def test_jupiter_phantom():
    """
    Generates 2D images and k-space data for different downsampling methods applied to the Jupiter Phantom. The results,
    including magnitude images and logarithmic-scaled k-space representations, are visualized and saved
    as an HTML file.
    """
    for i, shape in enumerate([(750, 600), (350, 300)]):
        jupiter = Phantom.get_jupiter(shape=shape)

        jupiter_img = jupiter.get_2d_image()
        jupiter_k_space = jupiter.get_2d_k_space()

        acceleration = 4
        ac_lines = 36

        images = [jupiter_img]
        k_s = [jupiter_k_space]
        names = ["fs", "us ac grappa", "us ac skip", "us ac weighted", "us ac random", "random"]
        sub_samplings = [
            jupiter.sub_sample_ac_grappa, jupiter.sub_sample_ac_skip_lines,
            jupiter.sub_sample_ac_weighted_lines, jupiter.sub_sample_ac_random_lines
        ]

        for sub_sample_func in sub_samplings:
            k = sub_sample_func(acceleration=acceleration, ac_lines=ac_lines)
            k_s.append(k)
            images.append(fft_to_img(k, dims=(0, 1)))

        k_s.append(jupiter.sub_sample_random(acceleration=acceleration, ac_central_radius=20))
        images.append(fft_to_img(k_s[-1], dims=(0, 1)))

        fig = psub.make_subplots(
            rows=2, cols=len(images),
            row_titles=["img", "k-space"],
            column_titles=names,
            shared_yaxes=True,
            shared_xaxes=True,
        )

        for idx, d in enumerate(images):
            fig.add_trace(
                go.Heatmap(z=torch.abs(d), transpose=True, colorscale="Viridis", showscale=False),
                row=1, col=idx + 1
            )
        for idx, d in enumerate(k_s):
            fig.add_trace(
                go.Heatmap(z=torch.log(torch.abs(d)), transpose=True, colorscale="Viridis", showscale=False),
                row=2, col=idx + 1
            )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        name_shape = f"{shape[0]}x{shape[1]}"
        output_dir = get_test_result_output_dir("phantom_creation", mode=ResultMode.VISUAL)
        fig.write_html(os.path.join(output_dir, f"jupiter_phantom_{name_shape}.html"))


def test_shepp_logan_phantom():
    """
    Generates 2D images and k-space data for different downsampling methods applied to the Shepp Logan Phantom.
    The results, including magnitude images and logarithmic-scaled k-space representations, are visualized and saved
    as an HTML file.
    """
    num_coils = 4
    num_echoes = 2
    shepp_logan = Phantom.get_shepp_logan(shape=(250, 200), num_coils=num_coils, num_echoes=num_echoes)

    shepp_logan_img = shepp_logan.get_2d_image()
    shepp_logan_k_space = shepp_logan.get_2d_k_space()

    acceleration = 4
    ac_lines = 36

    images = [shepp_logan_img]
    k_s = [shepp_logan_k_space]
    names = ["fs", "us ac grappa", "us ac skip", "us ac weighted", "us ac random", "random"]
    sub_samplings = [
        shepp_logan.sub_sample_ac_grappa, shepp_logan.sub_sample_ac_skip_lines,
        shepp_logan.sub_sample_ac_weighted_lines, shepp_logan.sub_sample_ac_random_lines
    ]

    for idx, s in enumerate(sub_samplings):
        k = s(acceleration=acceleration, ac_lines=ac_lines)
        k_s.append(k)
        images.append(fft_to_img(k, dims=(0, 1)))

    k_s.append(shepp_logan.sub_sample_random(acceleration=acceleration, ac_central_radius=20))
    images.append(fft_to_img(k_s[-1], dims=(0, 1)))

    fig = psub.make_subplots(
        rows=len(images), cols=2 + num_coils + num_echoes,
        row_titles=names,
        column_titles=["img c-1 e-1", "img rsos"] +
                      [f"c-{c + 1} e-1" for c in range(num_coils)] +
                      [f"c-1 e-{1 + e}" for e in range(num_echoes)],
        shared_yaxes=True,
        shared_xaxes=True,
    )

    for idx, d in enumerate(images):
        fig.add_trace(
            go.Heatmap(z=torch.abs(d[:, :, 0, 0]), transpose=False, colorscale="Viridis", showscale=False),
            row=1 + idx, col=1
        )
        fig.add_trace(
            go.Heatmap(z=torch.abs(root_sum_of_squares(d, dim_channel=-2)[:, :, 0]), transpose=False,
                       colorscale="Viridis", showscale=False),
            row=1 + idx, col=2
        )
    for idx, d in enumerate(k_s):
        for c in range(num_coils):
            fig.add_trace(
                go.Heatmap(
                    z=torch.log(torch.abs(d[:, :, c, 0])), transpose=False, colorscale="Viridis", showscale=False),
                row=1 + idx, col=3 + c
            )
        for e in range(num_echoes):
            fig.add_trace(
                go.Heatmap(
                    z=torch.log(torch.abs(d[:, :, 0, e])), transpose=False, colorscale="Viridis", showscale=False),
                row=1 + idx, col=3 + num_coils + e
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    output_dir = get_test_result_output_dir("phantom_creation", mode=ResultMode.VISUAL)
    fig.write_html(os.path.join(output_dir, "shepp_logan_phantom.html"))
