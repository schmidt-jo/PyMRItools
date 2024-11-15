import pathlib as plib
import logging

import torch
import plotly.graph_objects as go
import plotly.subplots as psub
import tqdm

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft, gaussian_2d_kernel, root_sum_of_squares
from pymritools.config.processing import DenoiseSettingsMPK

from pymritools.processing.denoising.mppca_k_filter.functions import matched_filter_noise_removal


def main():
    # set up phantom
    logging.info("get SheppLogan phantom")
    # build some coil sensitivities
    nc = 4
    shape = (256, 256)
    sl_img = SheppLogan().get_2D_image(shape=shape)

    # coil_sens = torch.zeros((*shape, 4))
    # centers = [[80, 80], [80, 200], [200, 80], [200, 200]]
    # for i in range(4):
    #     gw = gaussian_2d_kernel(size_x=shape[0], size_y=shape[1], center_x=centers[i][0],
    #                                             center_y=centers[i][1], sigma=(80, 100))
    #     gw /= torch.max(gw)
    #     coil_sens[:, :, i] = gw

    # build composite image
    # sl_cs_phantom = sl_img[:, :, None] * coil_sens * 100

    # convert to k_space
    sl_k = fft(input_data=sl_img, img_to_k=True, axes=(0, 1))[:,:,None]
    result = torch.zeros((6, 5, *shape))
    result[0, :, :, :] = sl_img[None].expand(5, -1, -1)

    for idx_n, n_scale in enumerate(tqdm.tqdm([5, 10, 20, 40, 60])):
        filt_k, filt_k_wo, k_noise, rem_noise, rem_noise_biased = denoise(n_scale=n_scale, k_space=sl_k)

        sl_img_recon = torch.squeeze(fft(input_data=k_noise, img_to_k=False, axes=(0, 1)))
        filt_img_recon = torch.squeeze(fft(input_data=filt_k, img_to_k=False, axes=(0, 1)))
        filt_img_recon_wo = torch.squeeze(fft(input_data=filt_k_wo, img_to_k=False, axes=(0, 1)))
        filt_rem_noise = torch.squeeze(fft(input_data=rem_noise, img_to_k=False, axes=(0, 1)))
        filt_rem_noise_biased = torch.squeeze(fft(input_data=rem_noise_biased, img_to_k=False, axes=(0, 1)))

        result[1, idx_n] = torch.abs(sl_img_recon)
        result[2, idx_n] = torch.abs(filt_img_recon)
        result[4, idx_n] = torch.abs(filt_img_recon_wo)
        result[3, idx_n] = torch.abs(filt_rem_noise)
        result[5, idx_n] = torch.abs(filt_rem_noise_biased)

    fig = psub.make_subplots(
        rows=6, cols=5,
        row_titles=["phantom.", "noisy image", "noise filtered image", "filtered noise", "bias intuition", "filtered bias"],
        horizontal_spacing=0.02, vertical_spacing=0.02
    )

    for idx_c in range(5):
        for idx_r in range(6):
            fig.add_trace(
                go.Heatmap(z=result[idx_r, idx_c], showscale=False), row=1+idx_r, col=idx_c + 1
            )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(width=1000, height=1200)
    fig_path = plib.Path("./examples/processing/denoising/results").absolute()
    file_name = fig_path.joinpath(f"mpk_tests").with_suffix(".html")
    logging.info(f"write file: {file_name}")
    fig.write_html(file_name)
    file_name = fig_path.joinpath(f"mpk_tests").with_suffix(".pdf")
    logging.info(f"write file: {file_name}")
    fig.write_image(file_name)
    print("done")


def denoise(n_scale: float, k_space: torch.Tensor):
    # add noise to k_space
    noise = torch.randn_like(k_space.real) + 1j * torch.randn_like(k_space.imag)
    noise *= torch.max(torch.abs(k_space)) / torch.max(torch.abs(noise)) /n_scale
    k_space_noise = k_space + noise

    # denoise
    settings = DenoiseSettingsMPK(
        out_path="./examples/processing/denoising/results",
        noise_mp_threshold=0.15, noise_mp_stretch=1.05
    )
    plib.Path(settings.out_path).absolute().joinpath("figs").mkdir(exist_ok=True, parents=True)

    logging.info(f"processing noise")
    # adopt dims -> noise [num_scans, num_channels, num_samples]
    noise = torch.movedim(noise[:100], -1, 1)
    input_filter = k_space_noise[:, :, None, :, None]
    filtered_k, rem_noise = matched_filter_noise_removal(
        noise_data_n_ch_samp=noise, k_space_lines_read_ph_sli_ch_t=input_filter, settings=settings
    )

    logging.info(f"processing w/o noise")
    # try to check what happens if we cut off singular values wo noise
    input_filter = k_space[:, :, None, :, None]
    filtered_k_wo_noise, rem_noise_biased = matched_filter_noise_removal(
        noise_data_n_ch_samp=noise, k_space_lines_read_ph_sli_ch_t=input_filter, settings=settings
    )
    return filtered_k, filtered_k_wo_noise, k_space_noise, rem_noise, rem_noise_biased


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    main()
