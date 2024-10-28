import pathlib as plib

import torch
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft, gaussian_2d_kernel, root_sum_of_squares
from pymritools.config.processing import DenoiseSettingsMPK

from pymritools.processing.denoising.mppca_k_filter.functions import matched_filter_noise_removal


def main():
    # set up phantom
    sl_phantom = SheppLogan()
    nx, ny = sl_phantom.shape
    # build some coil sensitivities
    nc = 4
    coil_sensitivities = torch.zeros((nx, ny, nc))

    # create
    for idx_c in range(nc):
        r = int(idx_c / 2)
        c = int(idx_c % 2)
        gw = gaussian_2d_kernel(
            size_x=nx, size_y=ny,
            center_x=80 + 240 * r, center_y=80 + 240 * c,
            sigma=160
        )
        gw /= torch.max(gw)
        coil_sensitivities[..., idx_c] = gw


    # build composite image
    sl_cs_phantom = sl_phantom[:, :, None] * coil_sensitivities

    # convert to k_space
    sl_k = fft(input_data=sl_cs_phantom, img_to_k=True, axes=(0, 1))

    # add noise to k_space
    noise = torch.randn_like(sl_k)
    noise *= torch.max(torch.abs(sl_k)) / torch.max(torch.abs(noise)) / 50
    sl_k_noise = sl_k + noise

    # denoise
    settings = DenoiseSettingsMPK(
        out_path="./examples/processing/denoising/results"
    )
    plib.Path(settings.out_path).absolute().joinpath("figs").mkdir(exist_ok=True, parents=True)

    # adopt dims -> noise [num_scans, num_channels, num_samples]
    noise = torch.movedim(noise[:10], -1, 1)
    input_filter = sl_k_noise[:, :, None, :, None]
    filtered_k = matched_filter_noise_removal(
        noise_data_n_ch_samp=noise, k_space_lines_read_ph_sli_ch_t=input_filter, settings=settings
    )

    # for plotting
    sl_img_recon = fft(input_data=sl_k_noise, img_to_k=False, axes=(0, 1))
    filt_img_recon = fft(input_data=filtered_k, img_to_k=False, axes=(0, 1))
    fig = psub.make_subplots(
        rows=3, cols=nc,
        row_titles=["phantom with coil sens.", "noisy recon image", "noise filtered image"]
    )
    for idx_c in range(nc):
        fig.add_trace(
            go.Heatmap(z=sl_cs_phantom[..., idx_c], showscale=False), row=1 , col=1 + idx_c
        )
        fig.add_trace(
            go.Heatmap(z=torch.abs(sl_img_recon[..., idx_c]), showscale=False), row=2, col=idx_c + 1
        )
        fig.add_trace(
            go.Heatmap(z=torch.abs(torch.abs(filt_img_recon[..., idx_c])), showscale=False), row=3, col=idx_c + 1
        )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    fig_path = plib.Path("./examples/processing/denoising/results").absolute()
    file_name = fig_path.joinpath("coil_sens").with_suffix(".html")
    fig.write_html(file_name)

    print("done")




if __name__ == '__main__':
    main()
