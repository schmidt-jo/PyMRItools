import d_mppca
import pathlib as plib
import nibabel as nib
import torch
import numpy as np
import logging
import typing
import tqdm
import plotly.express as px
from autodmri import estimator
log_module = logging.getLogger(__name__)


def denoise(
        input_path: str, save_path: str,
        fixed_p: int = 0, input_image_data: bool = True,
        noise_bias_correction: bool = True, noise_bias_mask: str = "",
        use_gpu: bool = True, gpu_device: int = 0, debug: bool = False
):
    conf = d_mppca.Config(
        in_path=input_path, save_path=save_path, fixed_p=fixed_p,
        noise_bias_correction=noise_bias_correction, noise_bias_mask=noise_bias_mask,
        input_image_data=input_image_data,
        use_gpu=use_gpu, gpu_device=gpu_device,
        debug=debug
    )
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    logging.info("___________________________________________________________________")
    logging.info("________________________ MP PCA Denoizing  ________________________")
    logging.info("___________________________________________________________________")

    # set logging level after possible config file read
    if conf.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        main(config=conf)
    except Exception as e:
        logging.exception(e)


def manjon_corr_model(gamma):
    if gamma < 1.86:
        return 0.0
    else:
        a = 0.9846 * (gamma - 1.86) + 0.1983
        b = gamma - 1.86 + 0.1175
        return a / b


def main(config: d_mppca.Config):
    file_path = plib.Path(config.in_path).absolute()
    if not file_path.is_file():
        err = f"file provided ({file_path.as_posix()}) not found"
        logging.error(err)
        raise FileNotFoundError(err)
    path = file_path.parent
    if config.save_path:
        save_path = plib.Path(config.save_path).absolute()
        save_path.mkdir(exist_ok=True, parents=True)
    else:
        save_path = path
    logging.info(f"load file: {file_path}")
    if ".nii" in file_path.suffixes:
        img = nib.load(file_path.as_posix())
        img_data = torch.from_numpy(img.get_fdata())
        affine = torch.from_numpy(img.affine)
    elif ".pt" in file_path.suffixes:
        img_data = torch.load(file_path.as_posix())
        af_path = plib.Path(config.in_affine).absolute()
        if not af_path.is_file():
            err = f"file provided ({af_path.as_posix()}) not found"
            logging.error(err)
            raise FileNotFoundError(err)
        affine = torch.load(af_path.as_posix())
    else:
        err = "file not .nii or .pt file"
        logging.error(err)
        raise AttributeError(err)
    stem = file_path.stem.split(".")[0]
    name = f"{config.file_prefix}_mppca"

    if config.use_gpu:
        device = torch.device(f"cuda:{config.gpu_device}")
    else:
        device = torch.device("cpu")

    # enable processing of coil combined data. Assume if input is 4D that we have a missing coil dim
    img_shape = img_data.shape
    if img_shape.__len__() < 4:
        msg = "assume no time dimension"
        log_module.info(msg)
        img_data = torch.unsqueeze(img_data, -1)
    if img_shape.__len__() < 5:
        msg = "assume no channel dimension"
        img_data = torch.unsqueeze(img_data, -2)
        img_shape = img_data.shape

    # we need to batch the data to fit on memory, easiest is to do it dimension based
    # want to batch channel dim and two slice axis (or any dim)
    # get vars
    nx, ny, nz, nch, m = img_shape
    cube_side_len = torch.ceil(torch.sqrt(torch.tensor([m]))).to(torch.int).item()
    n_v = cube_side_len ** 2
    ncx = nx - cube_side_len
    ncy = ny - cube_side_len
    # calculate const for mp inequality
    m_mp = min(m, n_v)
    if config.fixed_p > 0:
        p = config.fixed_p
        left_b = None
        right_a = None
        r_cumsum = None
        name += f"_fixed-p-{config.fixed_p}"
    else:
        p = None
        m_mp_arr = torch.arange(m_mp - 1)
        left_b = 4 * torch.sqrt(torch.tensor((m_mp - m_mp_arr) / n_v)).to(device=device, dtype=torch.float64)
        right_a = (1 / (m_mp - m_mp_arr)).to(device=device, dtype=torch.float64)
        # build a matrix to make the cummulative sum for the inequality calculation a matrix multiplication
        # dim [mmp, mmp - 1]
        r_cumsum = torch.triu(torch.ones(m_mp - 1, m_mp), diagonal=1).to(device=device, dtype=torch.float64)

    if not config.input_image_data:
        # if input is k-space data we convert to img space
        # loop over dim slices, batch dim channels
        logging.info(f"fft to image space")
        img_data = torch.fft.fftshift(
            torch.fft.fft2(
                torch.fft.ifftshift(
                    img_data,
                    dim=(0, 1)
                ),
                dim=(0, 1)
            ),
            dim=(0, 1)
        )
    # save max value to rescale later
    # if too high we set it to 1000
    max_val = torch.max(torch.abs(img_data))

    # we want to implement a first order stationary noise bias removal from Manjon 2015
    # with noise statistics from mask and St.Jean 2020
    if config.noise_bias_correction:
        mask_path = plib.Path(config.noise_bias_mask).absolute()
        if mask_path.is_file():
            nii_mask = nib.load(mask_path.as_posix())
            mask = nii_mask.get_fdata()
        else:
            msg = "no mask file provided, using autodmri to extract mask"
            log_module.info(msg)
            # use on first echo across all 3 dimensions
            # use rsos of channels if applicable, channel dim =-2
            input_data = torch.sqrt(
                torch.sum(
                    # take first echo and sum over channels
                    torch.abs(img_data[:, :, :, :, 0])**2,
                    dim=-1
                )
            )
            mask = np.ones(input_data.shape, dtype=bool)
            for idx_ax in tqdm.trange(3, desc="extracting noise voxels, autodmri"):
                _, _, tmp_mask = estimator.estimate_from_dwis(
                    data=torch.squeeze(input_data).numpy(), axis=idx_ax, return_mask=True, exclude_mask=None, ncores=16,
                    method='moments', verbose=0, fast_median=False
                )
                mask = np.bitwise_and(mask, tmp_mask.astype(bool))
            # save mask
            mask = mask.astype(np.int32)
            img = nib.Nifti1Image(mask, affine=affine.numpy())
            file_path = save_path.joinpath(f"autodmri_mask").with_suffix(".nii")
            logging.info(f"write file: {file_path.as_posix()}")
            nib.save(img, filename=file_path.as_posix())
        mask = torch.from_numpy(mask)
        # extend to time dim
        mask = mask[:, :, :, None, None].expand(-1, -1, -1, *img_data.shape[-2:]).to(torch.bool)
        # extract noise data
        noise_voxels = img_data[mask]
        noise_voxels = noise_voxels[noise_voxels > 0]
        sigma = get_sigma_from_noise_vox(noise_voxels)
        num_channels = torch.clamp(torch.round(get_n_from_noise_vox(noise_voxels, sigma)).to(torch.int), 1, 32)
        # save plot for reference
        noise_bins = torch.arange(int(max_val / 10)).to(noise_voxels.dtype)
        noise_hist, _ = torch.histogram(noise_voxels, bins=noise_bins, density=True)
        noise_hist /= torch.linalg.norm(noise_hist)
        noise_dist = noise_dist_ncc(noise_bins, sigma=sigma, n=num_channels)
        noise_dist /= torch.linalg.norm(noise_dist)
        noise_plot = torch.concatenate((noise_hist[:, None], noise_dist[:-1, None]), dim=1)

        name_list = ["noise voxels", f"noise dist. estimate, sigma: {sigma.item():.2f}, n: {num_channels.item()}"]
        fig = px.line(noise_plot, labels={'x': 'signal value [a,u,]', 'y': 'normalized count'})
        for i, trace in enumerate(fig.data):
            trace.update(name=name_list[i])
        fig_name = f"{name}_noise_histogramm"
        fig_file = save_path.joinpath(fig_name).with_suffix(".html")
        logging.info(f"write file: {fig_file.as_posix()}")
        fig.write_html(fig_file.as_posix())
        data_denoised_sq = torch.movedim(torch.zeros_like(img_data), (2, 3), (0, 1))
    else:
        sigma = None
        num_channels = None
        data_denoised_sq = None
    if max_val > 1e5:
        max_val = 1000
    if config.normalize:
        logging.info("normalize data, max 1 magnitude across time dimension")
        img_data = img_data / max_val
        name = f"{name}_normalized-input"
    img_data = torch.movedim(img_data, (2, 3), (0, 1))
    data_denoised = torch.zeros_like(img_data)
    data_n = torch.zeros((3, *img_data.shape))
    data_access = torch.zeros(data_denoised.shape[:-1], dtype=torch.float)
    data_p = torch.zeros(data_access.shape, dtype=torch.int)
    data_p_avg = torch.zeros_like(data_p)

    # correction factor (n_v~m)
    beta = 1.29

    logging.info(f"start processing")
    # x steps batched
    x_steps = torch.arange(ncx)[:, None] + torch.arange(cube_side_len)[None, :]
    for idx_y in tqdm.trange(img_data.shape[3] - cube_side_len, desc="loop over dim 1",
                             position=0, leave=False):
        patch = img_data[:, :, x_steps, idx_y:idx_y + cube_side_len].to(device)
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
        if config.fixed_p > 0:
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
        d = torch.matmul(torch.einsum("ijklm, ijkm -> ijklm", u, svs.to(img_data.dtype)), v)
        # manjon 2015: median of eigenvalues is related to local noise pattern
        # calculated from standard deviation, but we already subtracted patch mean, hence mean = 0.
        # keep only the ones lower than 2 * median std
        patch_evs = lam[torch.sqrt(lam) < 2 * torch.median(torch.sqrt(lam))]
        patch_sigma = beta * torch.sqrt(torch.median(patch_evs))
        local_snr = patch_loc_mean / patch_loc_std
        patch_sigma *= manjon_corr_model(local_snr)
        # add mean
        d += patch_mean
        # noise_sigma = (m / (m - 1)) * (torch.mean(d**2, dim=-1, keepdim=True) - torch.mean(d, dim=-1, keepdim=True)**2)
        # # we want to get the noise mean and std across the patch
        # noise_mean = torch.mean(noise, dim=-1, keepdim=True)
        # noise_std = torch.std(noise, dim=-1, keepdim=True)
        # shape back
        d = torch.movedim(d, -2, -1)
        d = torch.reshape(d, patch_shape).cpu()
        # noise_sigma = torch.movedim(noise_sigma, -2, -1)
        # noise_mean = torch.movedim(noise_mean, -2, -1)
        # noise_std = torch.movedim(noise_std, -2, -1)
        # # need to reverse reduction of spatial dims, dims [nz, nch, ncx, nv, m]
        # noise_sigma = noise_sigma.expand(-1, -1, -1, n_v, -1)
        # noise_sigma = torch.reshape(noise_sigma, patch_shape).cpu()
        # noise_mean = noise_mean.expand(-1, -1, -1, n_v, -1)
        # noise_mean = torch.reshape(noise_mean, patch_shape).cpu()
        # noise_std = noise_std.expand(-1, -1, -1, n_v, -1)
        # noise_std = torch.reshape(noise_std, patch_shape).cpu()

        # d = torch.movedim(d, 0, -2)
        # collect
        # dims [ch, z, c , c, m]
        # using the multipoint - pointwise approach of overlapping sliding window blocks from manjon et al. 2013
        # we summarize the contributions of each block at the relevant positions weighted by the
        # inverse number of nonzero coefficients in the diagonal eigenvalue / singular value matrix. i.e. P
        for idx_x in range(x_steps.shape[0]):
            data_denoised[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += (
                    theta_p[:, :, idx_x, None, None, None] * d[:, :, idx_x])
            # data_n[0, :, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += (
            #         theta_p[:, :, idx_x, None, None, None] * noise_sigma[:, :, idx_x])
            # data_n[1, :, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += (
            #         theta_p[:, :, idx_x, None, None, None] * noise_mean[:, :, idx_x])
            # data_n[2, :, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += (
            #         theta_p[:, :, idx_x, None, None, None] * noise_std[:, :, idx_x])
            data_access[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += theta_p[:, :, idx_x, None, None]
            data_p[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += num_p[:, :, idx_x, None, None]
            data_p_avg[:, :, x_steps[idx_x], idx_y:idx_y + cube_side_len] += 1
            if config.noise_bias_correction:
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
        data_n.real = torch.nan_to_num(
            torch.divide(data_n.real, data_access[None, :, :, :, :, None]),
            nan=0.0, posinf=0.0
        )
        data_n.imag = torch.nan_to_num(
            torch.divide(data_n.imag, data_access[None, :, :, :, :, None]),
            nan=0.0, posinf=0.0
        )
    else:
        data_denoised = torch.nan_to_num(
            torch.divide(data_denoised, data_access[:, :, :, :, None]),
            nan=0.0, posinf=0.0
        )
        data_n = torch.nan_to_num(
            torch.divide(data_n, data_access[None, :, :, :, :, None]),
            nan=0.0, posinf=0.0
        )
    # simple corrections scheme glenn - across time points versus diffusion gradients
    # sigma_glenn = (m / m-1) * (
    #         torch.mean(img_data**2, dim=-1, keepdim=True) - torch.mean(img_data, dim=-1, keepdim=True)**2
    # )
    if config.noise_bias_correction:
        # # original
        data_denoised_manjon = torch.sqrt(
            torch.clip(
                data_denoised ** 2 - 2 * num_channels * sigma ** 2,
                min=0.0
            )
        )
        # js
        # data_denoised_js_std = torch.sqrt(
        #     torch.clip(
        #         data_denoised ** 2 - 2 * num_channels * data_n[2]**2,
        #         min=0.0
        #     )
        # )
        # # glenn
        # data_denoised_glenn = torch.sqrt(
        #     torch.clip(
        #         data_denoised ** 2 - 2 * num_channels * sigma_glenn,
        #         min=0.0
        #     )
        # )
        # data_denoised_js_n = torch.sqrt(
        #     torch.clip(
        #         data_denoised ** 2 - 2 * num_channels * data_n[0],
        #         min=0.0
        #     )
        # )

    data_denoised = torch.movedim(data_denoised, (0, 1), (2, 3))
    data_n = torch.movedim(data_n, (1, 2), (3, 4))
    data_p_img = torch.nan_to_num(
        torch.divide(data_p, data_p_avg),
        nan=0.0, posinf=0.0
    )
    data_p_img = torch.movedim(data_p_img, (0, 1), (2, 3))

    if data_denoised.shape[-2] > 1:
        # [x, y, z, ch, t]
        data_denoised = torch.sqrt(
            torch.sum(
                torch.square(
                    torch.abs(
                        data_denoised
                    )
                ),
                dim=-2
            )
        )
        # [x, y, z, ch, t]
        data_n = torch.sqrt(
            torch.sum(
                torch.square(
                    torch.abs(
                        data_n
                    )
                ),
                dim=-2
            )
        )
        name += "_rsos"

    # save data
    data_denoised = torch.squeeze(data_denoised)
    data_n = torch.squeeze(data_n)
    name += f"_{stem}"
    data_denoised *= max_val / torch.max(data_denoised)

    img = nib.Nifti1Image(data_denoised.numpy(), affine=affine.numpy())
    file_path = save_path.joinpath(name).with_suffix(".nii")
    logging.info(f"write file: {file_path.as_posix()}")
    nib.save(img, filename=file_path.as_posix())
    # save access data
    img = nib.Nifti1Image(data_p_img.numpy(), affine=affine.numpy())
    file_path = save_path.joinpath(f"{name}_avg_p").with_suffix(".nii")
    logging.info(f"write file: {file_path.as_posix()}")
    nib.save(img, filename=file_path.as_posix())
    # save noise data
    noise_name = ["data_w-sum", "mean", "std"]
    for k in range(3):
        img = nib.Nifti1Image(data_n[k].numpy(), affine=affine.numpy())
        file_path = save_path.joinpath(f"{name}_noise_{noise_name[k]}").with_suffix(".nii")
        logging.info(f"write file: {file_path.as_posix()}")
        nib.save(img, filename=file_path.as_posix())

    file_name = save_path.joinpath(name).with_suffix(".pt")
    logging.info(f"write file: {file_name.as_posix()}")
    torch.save(data_denoised, file_name.as_posix())

    if config.noise_bias_correction:
        # names = ["manjon", "js_n", "js_std", "glenn"]
        names = ["manjon"]
        # ddata = [data_denoised_manjon, data_denoised_js_n, data_denoised_js_std, data_denoised_glenn]
        ddata = [data_denoised_manjon]
        for d_idx in range(len(names)):
            # save data
            dd = torch.movedim(ddata[d_idx], (0, 1), (2, 3))
            dd = torch.squeeze(dd)
            name_ = f"{name}_nbc-{names[d_idx]}"
            dd *= max_val / torch.max(dd)

            img = nib.Nifti1Image(dd.numpy(), affine=affine.numpy())
            file_path = save_path.joinpath(name_).with_suffix(".nii")
            logging.info(f"write file: {file_path.as_posix()}")
            nib.save(img, filename=file_path.as_posix())


def get_n_from_noise_vox(noise_voxel_data: torch.tensor, sigma: float):
    return 1 / (2 * noise_voxel_data.shape[0] * sigma ** 2) * torch.sum(noise_voxel_data ** 2, dim=0)


def get_sigma_from_noise_vox(noise_voxel_data: torch.tensor):
    num_pts = noise_voxel_data.shape[0]
    a = torch.sqrt(torch.tensor([1 / 2]))
    b = torch.sum(noise_voxel_data ** 4, dim=0) / torch.sum(noise_voxel_data ** 2, dim=0)
    c = 1 / num_pts * torch.sum(noise_voxel_data ** 2, dim=0)
    d = torch.sqrt(b - c)
    return a * d


def noise_dist_jean(x: torch.tensor, sigma: typing.Union[torch.tensor, float], n: typing.Union[torch.tensor, int]):
    sigma = torch.as_tensor(sigma).to(torch.float64)
    t = x ** 2 / (2 * sigma ** 2)
    n = torch.round(torch.as_tensor(n)).to(torch.int)
    return 1 / torch.exp(torch.lgamma(n)) * torch.pow(t, n - 1) * torch.exp(-t)


def noise_dist_ncc(x: torch.tensor, sigma: typing.Union[torch.tensor, float], n: typing.Union[torch.tensor, int]):
    sigma = torch.as_tensor(sigma).to(torch.float64)
    n = torch.round(torch.as_tensor(n)).to(torch.int)
    a = torch.pow(x, 2 * n - 1)
    b = torch.pow(2, n - 1) * torch.pow(sigma, 2 * n) * torch.exp(torch.lgamma(n))
    c = torch.exp((-x ** 2) / (2 * sigma ** 2))
    return a / b * c


def batch_cov_removed_mean(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def den_patch_mp(args):
    coll = []
    for item in args:
        patch, (idx, idy, idz), (m, n_v) = item
        patch_shape = patch.shape
        patch = np.reshape(patch, (-1, m))
        # do mean removed
        patch = patch - np.mean(patch, axis=0)
        # do svd
        u, s, v = np.linalg.svd(patch, full_matrices=False, compute_uv=True)

        # calculate inequality
        left = np.array([(s[p + 1] - s[m - 1]) / (4 * np.sqrt((m - p) / n_v)) for p in range(m - 1)])
        right = np.array([(1 / (m - p)) * np.sum(s[:p + 1]) for p in range(m - 1)])
        # minimum p for which left < right
        p = np.where(left < right)[0][0]

        # calculate denoised data
        d = np.matmul(u[:, :p], np.matmul(np.diag(s[:p]), v[:p]))
        # shape back
        d = np.reshape(d, patch_shape)
        # collect
        coll.append([d, (idx, idy, idz)])
    return coll
