"""
MPPCA denoising using method outlined in:
Does et al. 2019: Evaluation of principal component analysis image denoising on
multi‐exponential MRI relaxometry. Magn Reson Med
DOI: 10.1002/mrm.27658
_____
24.11.2023, Jochen Schmidt
"""
from pymritools.utils import nifti_load, nifti_save, fft, root_sum_of_squares, gaussian_2d_kernel
from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.processing.denoising import DenoiseSettings
from pymritools.processing.denoising.stats import non_central_chi as ncc_stats
import pathlib as plib

import torch
import numpy as np
import logging

import tqdm
import plotly.graph_objects as go
from autodmri import estimator
log_module = logging.getLogger(__name__)


def denoise(settings: DenoiseSettings):
    # load in data
    input_data, input_img = nifti_load(settings.in_path)
    input_data = torch.from_numpy(input_data)
    # ToDo: do for non .nii input

    # setup save path
    path_output = plib.Path(settings.out_path).absolute()
    if not path_output.exists():
        log_module.info(f"Setting up output path: {path_output.as_posix()}")
        path_output.mkdir(exist_ok=True, parents=True)

    if settings.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{settings.gpu_device}")
    else:
        device = torch.device("cpu")
    log_module.info(f"Configure device: {device}")

    # enable processing of coil combined data. Assume if input is 4D that we have a missing coil dim
    data_shape = input_data.shape
    msg = f"Found dimensions: {data_shape}."
    if data_shape.__len__() < 4:
        msg = f"{msg} Assume no time dimension -> adding dim."
        log_module.info(msg)
        input_data = input_data[..., None]
    if data_shape.__len__() < 5:
        msg = f"{msg} Assume no channel dimension -> adding dim. "
        input_data = torch.unsqueeze(input_data, -2)
    data_shape = input_data.shape
    if msg:
        logging.info(msg)
    del msg

    # we need to batch the data to fit on memory, easiest is to do it dimension based
    # want to batch channel dim and two slice axis (or any dim)
    # get vars
    nx, ny, nz, nch, m = data_shape
    cube_side_len = torch.ceil(torch.sqrt(torch.tensor([m]))).to(torch.int).item()
    n_v = cube_side_len ** 2
    ncx = nx - cube_side_len
    ncy = ny - cube_side_len
    # calculate const for mp inequality
    m_mp = min(m, n_v)
    if settings.fixed_p > 0:
        # no need to calculate thresholds
        p = settings.fixed_p
        log_module.info(f"Set fixed threshold p: {p}")
        left_b = None
        right_a = None
        r_cumsum = None
    else:
        # calculate threshold. Already do parts that are constant throughout to reduce overhead
        p = None
        m_mp_arr = torch.arange(m_mp - 1)
        left_b = 4 * torch.sqrt((m_mp - m_mp_arr) / n_v).to(device=device, dtype=torch.float64)
        right_a = (1 / (m_mp - m_mp_arr)).to(device=device, dtype=torch.float64)
        # build a matrix to make the cummulative sum for the inequality calculation a matrix multiplication
        # dim [mmp, mmp - 1]
        r_cumsum = torch.triu(torch.ones(m_mp - 1, m_mp), diagonal=1).to(device=device, dtype=torch.float64)

    if not settings.input_image_data:
        # if input is k-space data we convert to img space
        # loop over dim slices, batch dim channels
        logging.info(f"fft to image space")
        input_data = fft(input_data, inverse=False, axes=(0, 1))

    # save max value to rescale later
    # if too high we set it to 1000
    max_val = torch.max(torch.abs(input_data))

    # we want to implement a first order stationary noise bias removal from Manjon 2015
    # with noise statistics from mask and St.Jean 2020
    if settings.noise_bias_correction:
        mask_path = plib.Path(settings.noise_bias_mask).absolute()
        if mask_path.is_file():
            nii_mask, _ = nifti_load(mask_path.as_posix())
            mask = nii_mask.astype(np.int32)
        else:
            msg = "no mask file provided, using autodmri to extract mask"
            log_module.info(msg)
            # use on first echo across all 3 dimensions
            # use rsos of channels if applicable, channel dim =-2
            # take first echo and sum over channels
            input_data = root_sum_of_squares(input_data[..., 0], dim_channel=-1)

            mask = np.ones(input_data.shape, dtype=bool)
            # use autodmri to extract noise voxels
            for idx_ax in tqdm.trange(3, desc="extracting noise voxels, autodmri"):
                _, _, tmp_mask = estimator.estimate_from_dwis(
                    data=torch.squeeze(input_data).numpy(), axis=idx_ax, return_mask=True, exclude_mask=None, ncores=16,
                    method='moments', verbose=0, fast_median=False
                )
                mask = np.bitwise_and(mask, tmp_mask.astype(bool))
            # save mask
            mask = mask.astype(np.int32)
            nifti_save(data=mask, img_aff=input_img, path_to_dir=path_output, file_name=f"autodmri_mask")

        # get to torch
        mask = torch.from_numpy(mask)
        # extend to time dim
        mask = mask[:, :, :, None, None].expand(-1, -1, -1, *data_shape[-2:]).to(torch.bool)
        # extract noise data
        noise_voxels = input_data[mask]
        noise_voxels = noise_voxels[noise_voxels > 0]
        sigma, num_channels = ncc_stats.from_noise_voxels(noise_voxels)
        num_channels = torch.clamp(num_channels, 1, 32)

        # save plot for reference
        noise_bins = torch.arange(int(max_val / 10)).to(noise_voxels.dtype)
        noise_hist, _ = torch.histogram(noise_voxels, bins=noise_bins, density=True)
        noise_hist /= torch.linalg.norm(noise_hist)
        noise_dist = ncc_stats.noise_dist_ncc(noise_bins, sigma=sigma, n=num_channels)
        noise_dist /= torch.linalg.norm(noise_dist)
        noise_plot = torch.concatenate((noise_hist[:, None], noise_dist[:-1, None]), dim=1)

        # create plot
        fig = go.Figure()
        name_list = ["noise voxels", f"noise dist. estimate, sigma: {sigma.item():.2f}, n: {num_channels.item()}"]
        for idx_d, data in enumerate([noise_voxels, noise_hist]):
            fig.add_trace(
                go.Scattergl(
                    x=noise_bins, y=data, name=name_list[idx_d],
                )
            )
        fig.update_layout(
            title=go.layout.Title(
                text="Noise histogram",
            ),
            xaxis=dict(title='signal value [a,u,]'),
            yaxis=dict(title='normalized count')
        )

        fig_name = f"noise_histogramm"
        fig_file = path_output.joinpath(fig_name).with_suffix(".html")
        logging.info(f"write file: {fig_file.as_posix()}")
        fig.write_html(fig_file.as_posix())

        data_denoised_sq = torch.movedim(torch.zeros_like(input_data), (2, 3), (0, 1))
    else:
        sigma = None
        num_channels = None
        data_denoised_sq = None

    # scale
    if max_val > 1e5:
        max_val = 1000

    # reshuffle for batching data
    input_data = torch.movedim(input_data, (2, 3), (0, 1))
    data_denoised = torch.zeros_like(input_data)
    data_access = torch.zeros(data_denoised.shape[:-1], dtype=torch.float)
    data_p = torch.zeros(data_access.shape, dtype=torch.int)
    data_p_avg = torch.zeros_like(data_p)

    # correction factor (n_v~m)
    beta = 1.29

    logging.info(f"start processing")
    # x steps batched
    x_steps = torch.arange(ncx)[:, None] + torch.arange(cube_side_len)[None, :]

    for idx_y in tqdm.trange(data_shape[1] - cube_side_len, desc="loop over dim 1",
                             position=0, leave=False):
        patch = input_data[:, :, x_steps, idx_y:idx_y + cube_side_len].to(device)
        patch_shape = patch.shape
        patch = torch.reshape(patch, (nz, nch, ncx, -1, m))
        patch = torch.movedim(patch, -1, -2)
        # try batched svd
        # patch = img_data[:, :, start_x:end_x, start_y:end_y].to(device)
        # remove mean across spatial dim of patch
        patch_mean = torch.mean(patch, dim=-1, keepdim=True)
        patch_loc_mean = torch.mean(patch)
        patch_loc_std = torch.std(patch)
        patch -= patch_mean

        # do svd
        u, s, v = torch.linalg.svd(patch, full_matrices=False)
        # eigenvalues -> lambda = s**2 / n_v
        lam = s ** 2 / n_v
        svs = s.clone()
        if settings.fixed_p > 0:
            # we use the p first singular values
            svs[:, :, :, p:] = 0.0
            num_p = torch.full((nz, nch, ncx), p)
            theta_p = 1 / (1 + num_p)
        else:
            # calculate inequality, 3 batch dimensions!
            left = (lam[:, :, :, 1:] - lam[:, :, :, -1, None]) / left_b[None, None, None]
            r_lam = torch.einsum('is, czxs -> czxi', r_cumsum, lam)
            right = right_a[None, None, None] * r_lam
            # minimum p for which left < right
            # we actually find all p for which left < right and set those 0 in s
            p = left < right
            svs[:, :, :, :-1][p] = 0.0
            svs[:, :, :, -1] = 0.0
            num_p = torch.argmax(p.to(torch.int), dim=-1).cpu()
            theta_p = 1 / (1 + num_p.to(torch.float))
        # calculate denoised data, two batch dims!
        d = torch.matmul(torch.einsum("ijklm, ijkm -> ijklm", u, svs.to(input_data.dtype)), v)
        # manjon 2015: median of eigenvalues is related to local noise pattern
        # calculated from standard deviation, but we already subtracted patch mean, hence mean = 0.
        # keep only the ones lower than 2 * median std
        patch_evs = lam[torch.sqrt(lam) < 2 * torch.median(torch.sqrt(lam))]
        patch_sigma = beta * torch.sqrt(torch.median(patch_evs))
        local_snr = patch_loc_mean / patch_loc_std
        patch_sigma *= manjon_corr_model(local_snr)
        # add mean
        d += patch_mean

        # shape back
        d = torch.movedim(d, -2, -1)
        d = torch.reshape(d, patch_shape).cpu()

        # collect
        # dims [ch, z, c , c, m]
        # using the multipoint - pointwise approach of overlapping sliding window blocks from manjon et al. 2013
        # we summarize the contributions of each block at the relevant positions weighted by the
        # inverse number of nonzero coefficients in the diagonal eigenvalue / singular value matrix. i.e. P
        for idx_x in range(x_steps.shape[0]):
            data_denoised[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += (
                    theta_p[:, :, idx_x, None, None, None] * d[:, :, idx_x])
            data_access[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += theta_p[:, :, idx_x, None, None]
            data_p[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += num_p[:, :, idx_x, None, None]
            data_p_avg[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += 1
            if settings.noise_bias_correction:
                data_denoised_sq[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += (
                        theta_p[:, :, idx_x, None, None, None] * torch.abs(d[:, :, idx_x]) ** 2)

    if torch.is_complex(data_denoised):
        data_denoised.real = torch.nan_to_num(
            torch.divide(data_denoised.real, data_access[:, :, :, :, None]),
            nan=0.0, posinf=0.0
        )
        data_denoised.imag = torch.nan_to_num(
            torch.divide(data_denoised.imag, data_access[:, :, :, :, None]),
            nan=0.0, posinf=0.0
        )
    else:
        data_denoised = torch.nan_to_num(
            torch.divide(data_denoised, data_access[:, :, :, :, None]),
            nan=0.0, posinf=0.0
        )

    # simple corrections scheme glenn - across time points versus diffusion gradients
    # sigma_glenn = (m / m-1) * (
    #         torch.mean(input_data**2, dim=-1, keepdim=True) - torch.mean(img_data, dim=-1, keepdim=True)**2
    # )
    if settings.noise_bias_correction:
        # # original
        data_denoised_manjon = torch.sqrt(
            torch.clip(
                data_denoised ** 2 - 2 * num_channels * sigma ** 2,
                min=0.0
            )
        )
        # js
        # ToDo: check to subtract denoised image from original to see noise contributions.
        #  Use those noise contribution stats locally to bias correct

    # move dims back
    input_data = torch.movedim(input_data, (0, 1), (2, 3))
    data_denoised = torch.movedim(data_denoised, (0, 1), (2, 3))
    data_p_img = torch.nan_to_num(
        torch.divide(data_p, data_p_avg),
        nan=0.0, posinf=0.0
    )
    data_p_img = torch.movedim(data_p_img, (0, 1), (2, 3))

    # compute noise
    data_noise = input_data - data_denoised
    # compute variance and slightly smooth in xy dir (look for spatially dependent noise patterns)
    # for smoothing we use an fft kernel, after also taking the mean across the echoes
    kernel_size = torch.round(torch.tensor(data_shape[:2]).to(torch.float) / 20)
    window = gaussian_2d_kernel(
        size_x=nx, size_y=ny, sigma=kernel_size
    )
    data_noise_sm_var = torch.mean(data_noise ** 2, dim=-1)
    data_noise_sm_var = fft(data_noise_sm_var, inverse=True, axes=(0, 1))
    if not torch.is_complex(input_data):
        data_noise_sm_var = torch.real(data_noise_sm_var)
    data_noise_sm_var = data_noise_sm_var * window[:, :, None, None]
    data_noise_sm_var = fft(data_noise_sm_var, inverse=False, axes=(0, 1))
    if not torch.is_complex(input_data):
        data_noise_sm_var = torch.real(data_noise_sm_var)

    if data_denoised.shape[-2] > 1:
        # [x, y, z, ch, t]
        data_denoised = root_sum_of_squares(data_denoised, dim_channel=-2)

    # save data
    data_denoised = torch.squeeze(data_denoised)
    data_noise = torch.squeeze(data_noise)
    data_noise_sm_var = torch.squeeze(data_noise_sm_var)
    data_denoised *= max_val / torch.max(data_denoised)

    nifti_save(data=data_denoised.numpy(), img_aff=input_img, path_to_dir=path_output, file_name="denoised_data")
    nifti_save(data=data_noise.numpy(), img_aff=input_img, path_to_dir=path_output, file_name="noise_data")
    nifti_save(
        data=data_noise_sm_var.numpy(), img_aff=input_img, path_to_dir=path_output,
               file_name="noise_data_smoothed_var"
    )
    nifti_save(data=data_p_img.numpy(), img_aff=input_img, path_to_dir=path_output, file_name="avg_p")

    #
    # file_name = save_path.joinpath(name).with_suffix(".pt")
    # logging.info(f"write file: {file_name.as_posix()}")
    # torch.save(data_denoised, file_name.as_posix())

    if settings.noise_bias_correction:
        data_denoised_manjon = torch.movedim(data_denoised_manjon, (0, 1), (2, 3))

        nifti_save(
            data=data_denoised_manjon, img_aff=input_img, path_to_dir=path_output, file_name="denoised_data_nbc-manjon"
        )


def manjon_corr_model(gamma: float):
    if gamma < 1.86:
        return 0.0
    else:
        a = 0.9846 * (gamma - 1.86) + 0.1983
        b = gamma - 1.86 + 0.1175
        return a / b


def main():
    # set program logging
    setup_program_logging(name="MPPCA Denoising", level=logging.INFO)
    # set up argument parser
    parser, prog_args = setup_parser(
        prog_name="MPPCA Denoising",
        dict_config_dataclasses={"settings": DenoiseSettings}
    )
    # get settings
    settings = DenoiseSettings.from_cli(args=prog_args.settings, parser=parser)
    settings.display()

    try:
        denoise(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()
