import os.path

import torch

from pymritools.recon.loraks_dev.ps_loraks import Loraks, LowRankAlgorithmType
from pymritools.utils import Phantom, fft
from tests.utils import get_test_result_output_dir

import plotly.subplots as psub
import plotly.graph_objects as go


def run_ps_loraks():
    # set log level to info
    import logging
    logging.basicConfig(level=logging.INFO)
    loraks = (Loraks(max_num_iter=200)
              .with_device(torch.device("cuda"))
              .with_s_matrix()
              .with_torch_lowrank_algorithm(q=40, niter=2)
              # .with_sv_auto_soft_cutoff()
              .with_sv_hard_cutoff(40, 25)
              .with_patch_shape((5, 5))
              .with_sample_directions((1, 1))
              .with_linear_learning_rate(0.0001, 0.005)
              )

    # TODO: this shit needs to be fixed
    acceleration = 2
    k_shape = (512, 512)
    jupiter = Phantom.get_jupiter(k_shape)
    # jupiter_k_space = jupiter.sub_sample_random(acceleration)
    jupiter_k_space = jupiter.get_2d_k_space()
    mask = torch.randn_like(jupiter_k_space, dtype=torch.float32) > 0.8
    jupiter_k_space[mask] = 0.0
    jupiter_img = jupiter.get_2d_image()
    jupiter_k_test = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(jupiter_img)))
    jupiter_img_recon_test = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(jupiter_k_test)))
    jupiter_k_test[mask] = 0.0

    jupiter_img2 = fft(jupiter_k_space)

    jupiter_k_recon = loraks.reconstruct(jupiter_k_test[None], mask[None])
    jupiter_img_recon = fft(jupiter_k_recon[0])
    output_dir = get_test_result_output_dir("ps_loraks_quality")

    # Create Plotly figure with all datasets
    fig = psub.make_subplots(
        rows=2, cols=2,
        subplot_titles=("k-space (subsampled)", "Image (original)",
                        "k-space (reconstructed)", "Image (reconstructed)")
    )

    fig.add_trace(go.Heatmap(z=torch.log(torch.abs(jupiter_k_test)),
                             colorscale="Jet", showscale=False),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=torch.abs(jupiter_img_recon_test),
                             colorscale="Jet", showscale=False),
                  row=1, col=2)
    fig.add_trace(go.Heatmap(z=torch.log(torch.abs(jupiter_k_recon[0])),
                             colorscale="Jet", showscale=False),
                  row=2, col=1)
    fig.add_trace(go.Heatmap(z=torch.abs(jupiter_img_recon),
                             colorscale="Jet", showscale=False),
                  row=2, col=2)

    fig.update_layout(
        # height=800, width=800,
        title="PS-LORAKS Reconstruction Outputs",
        title_x=0.5
    )

    # Add these lines to ensure square aspect ratio for all subplots
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    # Optional: To remove axis labels and ticks for cleaner look
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)


    # Save figure as HTML
    fig.write_html(os.path.join(output_dir,"ps_loraks_reconstruction.html"))


if __name__ == '__main__':
    run_ps_loraks()
