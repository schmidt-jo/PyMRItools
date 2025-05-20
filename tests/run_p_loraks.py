import os.path

import torch

from pymritools.recon.loraks_dev_cleanup.p_loraks import PLoraks
from pymritools.utils import Phantom, fft_to_img
from tests.utils import get_test_result_output_dir

import plotly.subplots as psub
import plotly.graph_objects as go


def run_p_loraks():
    import logging
    logging.basicConfig(level=logging.INFO)
    loraks = (PLoraks()
              .with_regularization_lambda(0.3)
              .with_c_matrix()
              .with_torch_lowrank_algorithm(q=90, niter=2)
              .with_sv_auto_soft_cutoff()
              # .with_sv_hard_cutoff(20, 5)
              .with_patch_shape((10, 10))
              .with_sample_directions((1, 1))
              .with_linear_learning_rate(0.0001, 0.001)
              )

    acceleration = 3
    k_shape = (256, 256)
    jupiter = Phantom.get_shepp_logan(k_shape)
    jupiter_k = jupiter.get_2d_k_space().to(dtype=torch.complex64)
    jupiter_img_recon = fft_to_img(jupiter_k)
    jupiter_k_subsampled = jupiter.sub_sample_ac_random_lines(acceleration, 20).to(dtype=torch.complex64)
    jupiter_img_subsampled = fft_to_img(jupiter_k_subsampled)
    mask = torch.abs(jupiter_k_subsampled) > 10e-7

    jupiter_k_loraks_recon = loraks.reconstruct(jupiter_k_subsampled[None], mask[None])
    jupiter_img_loraks_recon = fft_to_img(jupiter_k_loraks_recon[0])
    output_dir = get_test_result_output_dir("ps_loraks_quality")

    fig = psub.make_subplots(
        rows=4, cols=2,
        subplot_titles=("k-space (original)", "Image (original)",
                        "k-space (subsampled)", "Image (subsampled)",
                        "k-space (reconstructed)", "Image (reconstructed)",
                        "k-space (difference)", "Image (difference)"
                        ),
        horizontal_spacing=0.01, vertical_spacing=0.05
    )

    fig.add_trace(go.Heatmap(z=torch.log(torch.abs(jupiter_k)),
                             colorscale="Viridis", showscale=False),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=torch.abs(jupiter_img_recon),
                             colorscale="Gray", showscale=False),
                  row=1, col=2)
    fig.add_trace(go.Heatmap(z=torch.log(torch.abs(jupiter_k_subsampled)),
                             colorscale="Viridis", showscale=False),
                  row=2, col=1)
    fig.add_trace(go.Heatmap(z=torch.sqrt(torch.abs(jupiter_img_subsampled)),
                             colorscale="Jet", showscale=False),
                  row=2, col=2)
    fig.add_trace(go.Heatmap(z=torch.log(torch.abs(jupiter_k_loraks_recon[0])),
                             colorscale="Viridis", showscale=False),
                  row=3, col=1)
    fig.add_trace(go.Heatmap(z=torch.sqrt(torch.abs(jupiter_img_loraks_recon)),
                             colorscale="Jet", showscale=False),
                  row=3, col=2)
    fig.add_trace(go.Heatmap(z=torch.log(torch.abs(jupiter_k_loraks_recon[0] - jupiter_k_subsampled)),
                             colorscale="Viridis", showscale=False),
                  row=4, col=1)
    fig.add_trace(go.Heatmap(z=torch.abs(jupiter_img_recon - jupiter_img_loraks_recon),
                             colorscale="Jet", showscale=False),
                  row=4, col=2)

    fig.update_layout(
        height=1600, width=800,
        title="PS-LORAKS Reconstruction Outputs",
        title_x=0.5
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.write_html(os.path.join(output_dir, "ps_loraks_reconstruction.html"))


if __name__ == '__main__':
    run_p_loraks()
