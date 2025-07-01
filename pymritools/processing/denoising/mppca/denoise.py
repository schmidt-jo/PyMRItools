import logging
import pathlib as plib

import torch
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
import tqdm
from scipy.ndimage import binary_erosion

from pymritools.config.processing import DenoiseSettingsMPPCA
from pymritools.utils import nifti_load, nifti_save, fft_to_img, ifft_to_k, root_sum_of_squares, torch_load
from pymritools.recon.loraks.matrix_indexing import get_linear_indices
from pymritools.processing.denoising.stats import non_central_chi as ncc_stats
from pymritools.config import setup_program_logging, setup_parser
from autodmri import estimator

log_module = logging.getLogger(__name__)


def load_data(settings: DenoiseSettingsMPPCA):
    # load in data
    path_in = plib.Path(settings.in_path)
    if ".nii" in path_in.suffixes:
        input_data, input_img = nifti_load(path_in)
        input_data = torch.from_numpy(input_data)
    elif ".pt" in path_in.suffixes:
        input_data = torch_load(path_in)
        if settings.in_affine:
            affine = torch_load(settings.in_affine)
        else:
            affine = torch.eye(4)
        input_img = nib.Nifti1Image(input_data.numpy(), affine.numpy())
    else:
        msg = f"Unsupported input file type ({path_in.suffixes})"
        raise AttributeError(msg)

    # setup save path
    path_output = plib.Path(settings.out_path).absolute()
    if not path_output.exists():
        log_module.info(f"Setting up output path: {path_output.as_posix()}")
        path_output.mkdir(exist_ok=True, parents=True)
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
    if msg:
        logging.info(msg)
    del msg

    if not settings.input_image_data:
        # if input is k-space data we convert to img space
        # loop over dim slices, batch dim channels
        logging.info(f"fft to image space")
        input_data = fft_to_img(input_data, dims=(0, 1))

    return input_data, input_img


def set_device(settings: DenoiseSettingsMPPCA):
    if settings.use_gpu and torch.cuda.is_available():
        device = torch.device(f"cuda:{settings.gpu_device}")
    else:
        device = torch.device("cpu")
    log_module.info(f"Configure device: {device}")
    return device


def set_constants(m: int, n_v: int, device, p: int = 0):
    # calculate const for mp inequality
    m_mp = min(m, n_v)
    if p > 0:
        # no need to calculate thresholds
        log_module.info(f"Set fixed threshold p: {p}")
        left_b = NotImplemented
        right_a = NotImplemented
        r_cumsum = NotImplemented
    else:
        # calculate threshold. Already do parts that are constant throughout to reduce overhead
        m_mp_arr = torch.arange(m_mp - 1)
        left_b = 4 * torch.sqrt((m_mp - m_mp_arr) / n_v).to(device=device, dtype=torch.float64)
        right_a = (1 / (m_mp - m_mp_arr)).to(device=device, dtype=torch.float64)
        # build a matrix to make the cummulative sum for the inequality calculation a matrix multiplication
        # dim [mmp, mmp - 1]
        r_cumsum = torch.triu(torch.ones(m_mp - 1, m_mp), diagonal=1).to(device=device, dtype=torch.float64)
    return right_a, left_b, r_cumsum


def extract_noise_stats_from_mask(
        input_data: torch.Tensor | np.ndarray, mask: torch.Tensor | np.ndarray,
        path_visuals: plib.Path | str = None):

    if not torch.is_tensor(input_data):
        input_data = torch.from_numpy(input_data)
    if not torch.is_tensor(mask):
        mask = torch.from_numpy(mask)

    data_shape = input_data.shape
    max_val = torch.max(torch.abs(input_data))

    while mask.shape.__len__() < len(data_shape):
        mask = mask[..., None]
    # Expand the mask to fit the data shape
    mask = mask.expand(*data_shape).to(torch.bool)
    # extract noise data
    noise_voxels = input_data[mask]
    noise_voxels = noise_voxels[noise_voxels > 0]
    sigma, num_channels = ncc_stats.from_noise_voxels(noise_voxels)
    num_channels = min(max(num_channels, 1), 32)

    # save plot for reference
    noise_bins = torch.arange(int(max_val / 10)).to(noise_voxels.dtype)
    noise_hist, _ = torch.histogram(noise_voxels, bins=noise_bins, density=True)
    noise_hist /= torch.linalg.norm(noise_hist)
    noise_dist = ncc_stats.noise_dist_ncc(noise_bins, sigma=sigma, n=num_channels)
    noise_dist /= torch.linalg.norm(noise_dist)

    if path_visuals is not None:
        path_vis = plib.Path(path_visuals).absolute()
        # create plot
        fig = go.Figure()
        name_list = ["noise voxels", f"noise dist. estimate, sigma: {sigma:.2f}, n: {num_channels}"]

        max_num_points = 10000
        if noise_voxels.shape[0] > max_num_points:
            plot_noise_vox = torch.permute(noise_voxels, dims=(0,))[:max_num_points]
        else:
            plot_noise_vox = noise_voxels
        fig.add_trace(
            go.Scattergl(
                x=plot_noise_vox, y=(0.02 * torch.randn_like(plot_noise_vox)) + 0.05,
                name="samples", mode="markers", marker=dict(size=2)
            )
        )

        for i, d in enumerate([noise_hist, noise_dist]):
            fig.add_trace(
                go.Scattergl(
                    x=noise_bins, y=d, name=name_list[i],
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
        fig_file = path_vis.joinpath(fig_name).with_suffix(".html")
        logging.info(f"write file: {fig_file.as_posix()}")
        fig.write_html(fig_file.as_posix())
    return sigma, num_channels


def extract_noise_mask(input_data: torch.Tensor, erode_iter: int = 2):
    # we want to implement a first order stationary noise bias removal from Manjon 2015
    # with noise statistics from mask and St.Jean 2020
    if not torch.is_tensor(input_data):
        err = f"Input data should be a torch tensor"
        raise AttributeError(err)

    msg = "using autodmri to extract mask"
    log_module.info(msg)
    # use echo across all 3 dimensions
    if input_data.shape.__len__() > 4:
        # use rsos of channels if applicable, channel dim =-2
        # take first echo and sum over channels
        admri_in = root_sum_of_squares(input_data, dim_channel=-2)
    else:
        admri_in = input_data

    mask = np.ones(admri_in.shape[:-1], dtype=bool)
    # use autodmri to extract noise voxels
    for idx_ax in tqdm.trange(3, desc="extracting noise voxels, autodmri"):
        _, _, tmp_mask = estimator.estimate_from_dwis(
            data=torch.squeeze(admri_in).numpy(), axis=idx_ax, return_mask=True, exclude_mask=None, ncores=16,
            method='moments', verbose=0, fast_median=False
        )
        mask = np.bitwise_and(mask, tmp_mask.astype(bool))
    # save mask
    mask = mask.astype(np.int32)
    # binary erode
    structure = np.zeros((3, 3, 3))
    sphere_ind = np.array(
        [[x, y, z] for x in range(3) for y in range(3) for z in range(3)
         if ((x - 1) ** 2 + (y - 1) ** 2 + (z - 1) ** 2) <= 1]
    )
    structure[sphere_ind[:, 0], sphere_ind[:, 1], sphere_ind[:, 2]] = 1
    for _ in range(erode_iter):
        mask = binary_erosion(mask, structure)
    # get to torch
    mask = torch.from_numpy(mask)
    return mask


def core_iteration(data_batch_b_nv_m: torch.Tensor,
                   right_a: torch.Tensor, left_b: torch.Tensor, r_cumsum: torch.Tensor, p: int = 0):
    # correction factor (n_v~m)
    beta = 1.29

    b, nv, m = data_batch_b_nv_m.shape
    # remove mean across spatial dim of patch
    patches_mean = torch.mean(data_batch_b_nv_m, dim=-1, keepdim=True)
    patches = data_batch_b_nv_m - patches_mean

    # do svd
    u, s, v = torch.linalg.svd(patches, full_matrices=False)
    # eigenvalues -> lambda = s**2 / n_v
    lam = s ** 2 / nv
    svs = s.clone()
    if p > 0:
        # we use the p first singular values
        svs[:, p:] = 0.0
        num_p = torch.full((b, nv), p, dtype=torch.int)
        theta_p = 1 / (1 + num_p)
    else:
        # calculate inequality, 1 batch dimensions!
        left = (lam[:, 1:] - lam[:, -1, None]) / left_b[None]
        r_lam = torch.einsum('is, bs -> bi', r_cumsum, lam)
        right = right_a[None] * r_lam
        # minimum p for which left < right
        # we actually find all p for which left < right and set those 0 in s
        p = left < right
        svs[:, :-1][p] = 0.0
        svs[:, -1] = 0.0
        num_p = torch.argmax(p.to(torch.int), dim=-1).cpu()
        # expand to patch
        num_p = num_p.unsqueeze(-1).expand(-1, nv).to(torch.int).contiguous()
        theta_p = 1 / (1 + num_p.to(torch.float))
    # calculate denoised data, one batch dims!
    d = torch.matmul(torch.einsum("ilm, im -> ilm", u, svs.to(u.dtype)), v)
    # manjon 2015: median of eigenvalues is related to local noise pattern
    # calculated from standard deviation, but we already subtracted patch mean, hence mean = 0.
    # keep only the ones lower than 2 * median std
    # patch_evs = lam[torch.sqrt(lam) < 2 * torch.median(torch.sqrt(lam))]
    # patch_sigma = beta * torch.sqrt(torch.median(patch_evs))
    # local_snr = patch_loc_mean / patch_loc_std
    # patch_sigma *= manjon_corr_model(local_snr)
    # add mean
    d += patches_mean
    return d, 2 * theta_p, num_p


def manjon_corr_model(gamma: float):
    if gamma < 1.86:
        return 0.0
    else:
        a = 0.9846 * (gamma - 1.86) + 0.1983
        b = gamma - 1.86 + 0.1175
        return a / b


def core_fn(
        input_data: torch.Tensor, p: int = 0, device: torch.device = torch.get_default_device()):
    # we ensure 5d data - take last dim as m for which we want to build the patches
    if input_data.shape.__len__() < 4:
        err = "Algorithm assumes at least 4D data, patches are build from slices (x-y) and 4th dimension (t)."
        raise AttributeError(err)
    if input_data.shape.__len__() < 5:
        msg = "No coil data detected, expanding coil dim to size 1."
        input_data = input_data.unsqueeze(-2)
    nx, ny, nz, nc, m = input_data.shape

    # get patch side length from m - we build patches from matrices of squared neighborhoods within the slice
    # (flattened) and the m dimension
    cube_side_len = torch.ceil(torch.sqrt(torch.tensor([m]))).to(torch.int).item()
    n_v = cube_side_len ** 2
    # calculate const for mp inequality - shorter side
    right_a, left_b, r_cumsum = set_constants(m=m, n_v=n_v, p=p, device=device)

    # batching nc, nz, and calculate batch_size dims to fit gpu
    input_data = input_data.view(nx, ny, -1, m)
    # move batch dim to front
    input_data = torch.movedim(input_data, 2, 0)
    # get indices
    data_batched_shape = input_data.shape
    indices, matrix_shape = get_linear_indices(
        k_space_shape=data_batched_shape[1:], patch_shape=(cube_side_len, cube_side_len, -1),
        sample_directions=(1, 1, 0)
    )
    indices_b, _ = get_linear_indices(
        k_space_shape=data_batched_shape[1:-1], patch_shape=(cube_side_len, cube_side_len),
        sample_directions=(1, 1)
    )
    # still want m dim to be last dim
    nxy, nb = matrix_shape
    nb = nb // m
    matrix_shape = (nxy, nb, m)

    # allocate data
    data_denoised = torch.zeros(data_batched_shape, dtype=input_data.dtype).view(data_batched_shape[0], -1)
    data_access = torch.zeros(data_batched_shape[:-1], dtype=torch.float).view(data_batched_shape[0], -1)
    data_p = torch.zeros_like(data_access, dtype=torch.int).view(data_batched_shape[0], -1)

    # data_p_avg = torch.zeros_like(data_p)
    for si, s in enumerate(tqdm.tqdm(input_data, desc="Batch denoising")):
        data_b_nv_m = s.flatten()[indices].view(matrix_shape).to(device)
        d, theta_p, num_p = core_iteration(
            data_batch_b_nv_m=data_b_nv_m, p=p, right_a=right_a, left_b=left_b, r_cumsum=r_cumsum
        )
        data_denoised[si] = data_denoised[si].view(-1).index_add(0, indices, d.view(-1).cpu())
        data_access[si] = data_access[si].view(-1).index_add(0, indices_b, theta_p.view(-1).cpu())
        data_p[si] = data_p[si].view(-1).index_add(0, indices_b, num_p.view(-1).cpu())
    # reshape
    data_denoised = data_denoised.view(data_batched_shape)
    data_access = data_access.view(data_batched_shape[:-1])
    data_p = data_p.view(data_batched_shape[:-1])
    # move batch dim back
    data_denoised = torch.movedim(data_denoised, 0, 2)
    data_access = torch.movedim(data_access, 0, 2)
    data_p = torch.movedim(data_p, 0, 2)
    # reshape
    data_denoised = data_denoised.view(nx, ny, nz, nc, m)
    data_access = data_access.view(nx, ny, nz, nc)
    data_p = data_p.view(nx, ny, nz, nc)
    # avg
    data_access[data_access < 1e-5] = 1
    if torch.is_complex(data_denoised):
        data_denoised.real = torch.divide(
            data_denoised.real, data_access[:, :, :, :, None]
        )
        data_denoised.imag = torch.divide(
            data_denoised.imag, data_access[:, :, :, :, None]
        )
    else:
        data_denoised = torch.divide(
            data_denoised, data_access[:, :, :, :, None]
        )
    return torch.squeeze(data_denoised), torch.squeeze(data_access), torch.squeeze(data_p)


def noise_bias_correction(
        denoised_data: torch.Tensor, sigma, num_channels):
    # correct
    return torch.sqrt(
        torch.clip(
            denoised_data ** 2 - 2 * num_channels * sigma ** 2,
            min=0.0
        )
    )


def get_noise_mask(input_data: torch.Tensor, mask_path: plib.Path | str = None):
    if mask_path is not None:
        mask_path = plib.Path(mask_path)
        if not mask_path.is_file():
            err = f"Mask file {mask_path} does not exist"
            raise FileNotFoundError(err)
        nii_mask, _ = nifti_load(mask_path.as_posix())
        mask = nii_mask.astype(np.int32)
    else:
        mask = extract_noise_mask(input_data=input_data)
    return mask


def denoise(settings: DenoiseSettingsMPPCA):
    # load data
    input_data, input_img = load_data(settings=settings)

    # set device
    device = set_device(settings=settings)

    data_denoised, data_access, data_p = core_fn(input_data=input_data, p=settings.fixed_p, device=device)

    if settings.noise_bias_correction:
        # get noise mask
        mask = get_noise_mask(input_data=input_data, mask_path=settings.noise_bias_mask)

        # check visuals
        if settings.visualize:
            v_fn = plib.Path(settings.out_path)
        else:
            v_fn = None

        # get noise stats
        sigma, num_channels = extract_noise_stats_from_mask(
            input_data=input_data, mask=mask,
            path_visuals=v_fn
        )

        data_denoised_nbc = noise_bias_correction(
            denoised_data=data_denoised, sigma=sigma, num_channels=num_channels
        )
    else:
        data_denoised_nbc = None
        mask = None

    # compute noise
    data_noise = torch.squeeze(input_data) - data_denoised

    save_data(
        data_denoised=data_denoised, data_noise=data_noise, noise_mask=mask, data_p=data_p,
        data_denoised_nbc=data_denoised_nbc, nii_img=input_img, out_path=settings.out_path
    )


def save_data(
        data_denoised: torch.Tensor, data_noise: torch.Tensor, data_p: torch.Tensor,
        nii_img: nib.Nifti1Image, out_path: plib.Path | str,
        noise_mask: torch.Tensor = None, data_denoised_nbc: torch.Tensor = None):
    if data_denoised.shape[-2] > 1:
        # [x, y, z, ch, t]
        data_denoised = root_sum_of_squares(data_denoised, dim_channel=-2)
    path_output = plib.Path(out_path).absolute()

    # save data
    data_denoised = torch.squeeze(data_denoised)
    data_noise = torch.squeeze(data_noise)
    # data_noise_sm_var = torch.squeeze(data_noise_sm_var)
    # data_denoised *= max_val / torch.max(data_denoised)

    nifti_save(data=data_denoised.numpy(), img_aff=nii_img, path_to_dir=path_output, file_name="denoised_data")
    nifti_save(data=data_noise.abs().numpy(), img_aff=nii_img, path_to_dir=path_output, file_name="noise_data")
    # nifti_save(
    #     data=data_noise_sm_var.numpy(), img_aff=nii_img, path_to_dir=path_output,
    #            file_name="noise_data_smoothed_var"
    # )
    nifti_save(data=data_p.numpy(), img_aff=nii_img, path_to_dir=path_output, file_name="avg_p")

    #
    # file_name = save_path.joinpath(name).with_suffix(".pt")
    # logging.info(f"write file: {file_name.as_posix()}")
    # torch.save(data_denoised, file_name.as_posix())

    if data_denoised_nbc is not None:
        if noise_mask is not None:
            nifti_save(
                data=noise_mask.to(torch.int32), img_aff=nii_img, path_to_dir=path_output, file_name="noise_mask"
            )
        nifti_save(
            data=data_denoised_nbc, img_aff=nii_img, path_to_dir=path_output, file_name="denoised_data_nbc"
        )


def main():
    # set program logging
    setup_program_logging(name="MPPCA Denoising", level=logging.INFO)
    # set up argument parser
    parser, prog_args = setup_parser(
        prog_name="MPPCA Denoising",
        dict_config_dataclasses={"settings": DenoiseSettingsMPPCA}
    )
    # get settings
    # prog_args.settings.out_path = "/data/pt_np-jschmidt/code/PyMRItools/examples/processing/denoising/results"
    settings = DenoiseSettingsMPPCA.from_cli(args=prog_args.settings, parser=parser)
    settings.display()

    try:
        denoise(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()
