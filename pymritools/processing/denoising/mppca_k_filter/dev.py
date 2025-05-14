import pathlib as plib
import logging

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as psub
import tqdm

from pymritools.utils.phantom import SheppLogan
from pymritools.utils import fft_to_img, ifft_to_k, gaussian_2d_kernel, root_sum_of_squares
from pymritools.config.processing import DenoiseSettingsMPK

from pymritools.processing.denoising.mppca_k_filter.functions import (
    matched_filter_noise_removal, distribution_mp, interpolate
)


def main():
    fig_path = plib.Path("./examples/processing/denoising/results").absolute()
    # set up phantom
    logging.info("get SheppLogan phantom")
    # build some coil sensitivities
    nc = 32
    shape = (256, 256)
    sl_img = SheppLogan().get_2D_image(shape=shape)


    coil_sens = torch.zeros((*shape, nc))
    for i in range(nc):
        center_x = torch.randint(low=20, high=shape[0] - 20, size=(1,))
        center_y = torch.randint(low=20, high=shape[1] - 20, size=(1,))
        gw = gaussian_2d_kernel(
            size_x=shape[0], size_y=shape[1],
            center_x=center_x.item(), center_y=center_y.item(), sigma=(40, 60)
        )
        gw /= torch.max(gw)
        coil_sens[:, :, i] = gw

    # build composite image dims [nx, ny, nch]
    sl_cs_phantom = sl_img[:, :, None] * coil_sens * 100
    sl_k_cs = fft_to_img(input_data=sl_cs_phantom, dims=(0, 1))

    fig = psub.make_subplots(
        rows=4, cols=nc // 4,
        horizontal_spacing=0.02, vertical_spacing=0.02
    )

    for idx_c in range(nc // 4):
        for idx_r in range(4):
            idx = idx_c * 4 + idx_r
            fig.add_trace(
                go.Heatmap(z=torch.abs(sl_cs_phantom[:, :, idx]), showscale=False),
                row=1+idx_r, col=idx_c + 1
            )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(width=1000, height=1200)
    file_name = fig_path.joinpath(f"coil_sens").with_suffix(".html")
    logging.info(f"write file: {file_name}")
    fig.write_html(file_name)
    file_name = fig_path.joinpath(f"coil_sens").with_suffix(".pdf")
    logging.info(f"write file: {file_name}")
    fig.write_image(file_name)

    # convert to k_space
    sl_k = fft_to_img(input_data=sl_img, dims=(0, 1))[:, :, None]

    snrs = [20, 10, 5, 3, 2]
    # want to plot some rows (different denoising stages) and some cols (different snrs) for some data variants
    # take here rsos mpk, ch1 mpk, ch2 mpk, rsos mpch, ch1, mpch, ch2 mpch
    # for each we want to calculate a noisy image, filtered image, filtered noise, bias,
    # and filtered bias image (6 with ground truth)
    # those are dim 1, dim 2 is each snr
    plot_data_mpk_rsos = torch.zeros((6, len(snrs), *shape))
    plot_data_mpk_rsos[0] = root_sum_of_squares(sl_cs_phantom[None].expand(5, -1, -1, -1), dim_channel=-1)

    plot_data_mpk_ch1 = torch.zeros((6, len(snrs), *shape))
    plot_data_mpk_ch1[0] = sl_cs_phantom[None].expand(5, -1, -1, -1)[:, :, :, 0]

    plot_data_mpk_ch2 = torch.zeros((6, len(snrs), *shape))
    plot_data_mpk_ch2[0] = sl_cs_phantom[None].expand(5, -1, -1, -1)[:, :, :, 1]

    plot_data_mpch_rsos = torch.zeros((6, len(snrs), *shape))
    plot_data_mpch_rsos[0] = root_sum_of_squares(sl_cs_phantom[None].expand(5, -1, -1, -1), dim_channel=-1)

    plot_data_mpch_ch1 = torch.zeros((6, len(snrs), *shape))
    plot_data_mpch_ch1[0] = sl_cs_phantom[None].expand(5, -1, -1, -1)[:, :, :, 0]

    plot_data_mpch_ch2 = torch.zeros((6, len(snrs), *shape))
    plot_data_mpch_ch2[0] = sl_cs_phantom[None].expand(5, -1, -1, -1)[:, :, :, 1]

    for idx_n, snr in enumerate(tqdm.tqdm(snrs)):
        # do the denoising for all channels
        filt_k, filt_k_wo, k_noise, rem_noise, rem_noise_biased = denoise_mpk(snr=snr, k_space=sl_k_cs)
        ch_filt_k, ch_filt_k_wo, ch_k_noise, ch_rem_noise, ch_rem_noise_biased = denoise_mpchannels(
            snr=snr, k_space=sl_k_cs
        )
        # add to plot data
        for idx_d, d in enumerate([k_noise, filt_k, rem_noise, filt_k_wo, rem_noise_biased]):
            # rSoS of mpk
            plot_data_mpk_rsos[1+idx_d, idx_n] = root_sum_of_squares(
                torch.squeeze(ifft_to_k(input_data=d, dims=(0, 1))), dim_channel=-1
            )
            # channel 1
            plot_data_mpk_ch1[1+idx_d, idx_n] = torch.abs(torch.squeeze(
                ifft_to_k(input_data=d, dims=(0, 1))
            ))[:, :, 0]
            # channel 2
            plot_data_mpk_ch2[1+idx_d, idx_n] = torch.abs(torch.squeeze(
                ifft_to_k(input_data=d, dims=(0, 1))
            ))[:, :, 1]
        for idx_d, d in enumerate([ch_k_noise, ch_filt_k, ch_rem_noise, ch_filt_k_wo, ch_rem_noise_biased]):
            # rSoS of mpch
            plot_data_mpch_rsos[1+idx_d, idx_n] = root_sum_of_squares(
                torch.squeeze(ifft_to_k(input_data=d, dims=(0, 1))), dim_channel=-1
            )
            # channel 1
            plot_data_mpch_ch1[1+idx_d, idx_n] = torch.abs(torch.squeeze(
                ifft_to_k(input_data=d, dims=(0, 1))
            ))[:, :, 0]
            # channel 2
            plot_data_mpch_ch2[1+idx_d, idx_n] = torch.abs(torch.squeeze(
                ifft_to_k(input_data=d, dims=(0, 1))
            ))[:, :, 1]


    names = ["MPK rsos", "MPCh rsos", "MPK Ch. 1", "MPCh Ch. 1", "MPK Ch. 2", "MPCh Ch. 2"]
    for idx_d, data in enumerate([
        plot_data_mpk_rsos, plot_data_mpch_rsos, plot_data_mpk_ch1, plot_data_mpch_ch1,
        plot_data_mpk_ch2, plot_data_mpch_ch2
    ]):
        fig = psub.make_subplots(
            rows=len(names), cols=5,
            row_titles=["phantom.", "noisy image", "noise filtered image", "filtered noise", "bias intuition", "filtered bias"],
            column_titles=[f"SNR: {s}" for s in snrs],
            horizontal_spacing=0.02, vertical_spacing=0.02
        )
        # each of the data is dims [6, snrs, shape]
        for idx_r, rd in enumerate(data):
            for idx_c in range(len(snrs)):
                fig.add_trace(
                    go.Heatmap(z=rd[idx_c], showscale=False), row=1+idx_r, col=idx_c + 1
                )
        fig.update_yaxes(visible=False)
        fig.update_xaxes(visible=False)
        fig.update_layout(width=1000, height=1200)
        file_name = fig_path.joinpath(names[idx_d]).with_suffix(".html")
        logging.info(f"write file: {file_name}")
        fig.write_html(file_name)
        file_name = fig_path.joinpath(names[idx_d]).with_suffix(".pdf")
        logging.info(f"write file: {file_name}")
        fig.write_image(file_name)
    fig = psub.make_subplots(
        rows=6, cols=5,
        row_titles=["phantom.", "noisy image", "noise filtered image", "filtered noise", "bias intuition", "filtered bias"],
        column_titles=[f"SNR: {s}" for s in snrs],
        horizontal_spacing=0.02, vertical_spacing=0.02
    )
    # each of the data is dims [6, snrs, shape]
    for idx_r, rd in enumerate(data):
        for idx_c in range(len(snrs)):
            fig.add_trace(
                go.Heatmap(z=rd[idx_c], showscale=False), row=1+idx_r, col=idx_c + 1
            )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(visible=False)
    fig.update_layout(width=1000, height=1200)
    file_name = fig_path.joinpath(names[idx_d]).with_suffix(".html")
    logging.info(f"write file: {file_name}")
    fig.write_html(file_name)
    file_name = fig_path.joinpath(names[idx_d]).with_suffix(".pdf")
    logging.info(f"write file: {file_name}")
    fig.write_image(file_name)



def denoise_mpk(snr: float, k_space: torch.Tensor):
    # add noise to k_space
    noise = torch.randn_like(k_space)
    # noise has mean 0 and var 1
    # want to set the SNR via max(data) / var noise = 1; hence set the max to snr
    k_space_noise = k_space * 4 * torch.pi / torch.max(torch.abs(k_space)) * snr + noise

    # denoise
    settings = DenoiseSettingsMPK(
        out_path="./examples/processing/denoising/results",
        noise_mp_threshold=0.15, noise_mp_stretch=1.05, batch_size=300
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


def denoise_mpchannels(snr, k_space: torch.Tensor):
    # add noise to k_space
    noise = torch.randn_like(k_space)
    # noise has mean 0 and var 1
    # want to set the SNR via max(data) / var noise = 1; hence set the max to snr
    k_space_noise = k_space * 4 * torch.pi / torch.max(torch.abs(k_space)) * snr + noise

    # want to extract the distribution from the noise
    hist_depth = 200
    # reshape noise to have channels at last dim
    noise = noise[0]
    n, m = noise.shape
    # get the eigenvalues
    svals = torch.linalg.svdvals(noise)
    s_lam = svals**2 / n

    # take some percentage bigger than biggest eigenvalue
    noise_s_max = 1.2 * torch.max(s_lam)
    # build some axis for which we calculate the mp - distribution
    noise_ax = torch.linspace(0, noise_s_max, hist_depth, dtype=s_lam.dtype)

    gamma = m / n
    # get biggest and lowest s to approximate noise characteristics from
    sigma = torch.sqrt(
        (1.1 * torch.max(s_lam, dim=0).values - torch.min(s_lam, dim=0).values) /
        4 / np.sqrt(gamma)
    )
    if len(sigma.shape) == 0:
        sigma = sigma.unsqueeze(0)

    # get mp distribution of noise values for all channels
    p_noise = distribution_mp(noise_ax, sigma, gamma)
    p_noise /= torch.sum(p_noise, dim=len(sigma.shape), keepdim=True)

    # We want to convert this distribution to a weighting.
    # ie: the more probable singular values of our data are in this distribution,
    # the more we want to threshold them
    # do some adjustments to convert to weighting factor. This needs some testing of optimality!
    p_noise_w = torch.clamp(
        p_noise / torch.max(p_noise, dim=len(sigma.shape), keepdim=True).values,
        0, 0.2
    )
    # scale back to 1
    p_noise_w /= torch.max(p_noise_w, dim=len(sigma.shape), keepdim=True).values
    # fill in ones in front
    p_noise_w[:, :int(hist_depth / 10)] = 1

    # invert distribution to create weighting
    p_weight = (1 - p_noise_w / torch.max(p_noise_w, dim=len(sigma.shape), keepdim=True).values)
    p_weight_ax = noise_ax

    # take first dim as batch dim, save dims and compute batch dim, dims [x, y, ch]
    _, n, m = k_space.shape
    # dims [x, min(y, ch)]
    u, s, vh = torch.linalg.svd(k_space_noise, full_matrices=False)
    bs_eigv = s ** 2 / n
    # we combined the channels, need to unsqueeze one channel dim
    bs_eigv = bs_eigv.unsqueeze(1)

    # can calculate the weighting for the whole batched singular values at once
    weighting = interpolate(x=bs_eigv, xp=p_weight_ax, fp=p_weight)
    # we could also try original idea: make histogram and find / correlate the noise
    # MP-distribution in the histogram, then remove, for now we just weight

    # weight original singular values (not eigenvalues)
    s_w = s * torch.squeeze(weighting)
    # reconstruct signal with filtered singular values, dims [batch, m, channels]
    signal_filt = torch.matmul(
        torch.einsum("ilm, im -> ilm", u, s_w.to(u.dtype)),
        vh)
    # normalize to previous signal levels - taking maximum of absolute value across channels
    # assign and move off GPU
    # k_space_filt[start:end] = signal_filt.cpu()
    # filtered_noise[start:end] = (batch - signal_filt).cpu()

    # reconstruct noise
    noise_filt = k_space_noise - signal_filt

    # redo with k_space no noise
    # dims [x, min(y, ch)]
    u, s, vh = torch.linalg.svd(k_space, full_matrices=False)
    bs_eigv = s ** 2 / n
    # we combined the channels, need to unsqueeze one channel dim
    bs_eigv = bs_eigv.unsqueeze(1)

    # can calculate the weighting for the whole batched singular values at once
    weighting = interpolate(x=bs_eigv, xp=p_weight_ax, fp=p_weight)
    # we could also try original idea: make histogram and find / correlate the noise
    # MP-distribution in the histogram, then remove, for now we just weight

    # weight original singular values (not eigenvalues)
    s_w = s * torch.squeeze(weighting)
    # reconstruct signal with filtered singular values, dims [batch, m, channels]
    signal_no_noise_filt =  torch.matmul(
        torch.einsum("ilm, im -> ilm", u, s_w.to(u.dtype)),
        vh
    )

    noise_no_noise_filt = k_space - signal_no_noise_filt

    return signal_filt, signal_no_noise_filt, k_space_noise, noise_filt, noise_no_noise_filt



if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )
    main()
