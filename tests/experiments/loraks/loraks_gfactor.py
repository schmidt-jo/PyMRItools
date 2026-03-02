import pathlib as plib
import logging

import torch
import numpy as np

import plotly.graph_objects as go
import plotly.subplots as psub
from git.util import join_path

from pymritools.config.basic import setup_program_logging
from pymritools.utils import torch_load, torch_save, fft_to_img, ifft_to_k, Phantom

from pymritools.recon.loraks.loraks import Loraks, LoraksBase
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions
from pymritools.recon.loraks.utils import (
    check_channel_batch_size_and_batch_channels, prepare_k_space_to_batches, unprepare_batches_to_k_space,
    pad_input, unpad_output
)
from pymritools.modeling.dictionary.grid_search_channels import smooth_map
from pymritools.processing.denoising.lcpca import denoise_lcpca
from tests.utils import get_test_result_output_dir, ResultMode

logger = logging.getLogger(__name__)


def plot_data(img_in: torch.Tensor, path: plib.Path, name: str, zmax=None, zmin=None):
    logging.info("Plot data")
    if img_in.ndim < 5:
        img_in = img_in.unsqueeze(-1)
    num_channels = img_in.shape[-2]
    num_imgs = img_in.shape[2]
    fig = psub.make_subplots(rows=num_imgs, cols=min(num_channels, 5))
    for i, d in enumerate(img_in.permute(2, 0, 1, 3, 4)):
        d = d[..., 15:20, 0] if num_channels > 20 else d[..., :, 0]
        for j, dn in enumerate(d.permute(2, 0, 1)):
            fig.add_trace(
                go.Heatmap(
                    z=dn.abs(),
                    zmin=0 if zmin is None else zmin[i],
                    zmax=None if zmax is None else zmax[i],
                    transpose=True, showscale=False, colorscale="Inferno"
                ),
                row=i + 1, col=j + 1)
    fn = path.joinpath(name).with_suffix(".html").as_posix()
    logging.info(f"Write file: {fn}")
    fig.write_html(fn)

def loraks_recon(k: torch.Tensor, loraks: LoraksBase, batch_size_channels: int = 8):
    logger.info("Prepare Data")
    # batching
    batch_channel_indices = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k, batch_size_channels=batch_size_channels
    )
    k_batched, input_shape = prepare_k_space_to_batches(
        k_space_rpsct=k, batch_channel_indices=batch_channel_indices
    )
    # padding
    k_batched, padding = pad_input(k_batched)

    logger.info("Reconstruction")
    k_recon = loraks.reconstruct(k_batched)
    torch.cuda.empty_cache()

    logger.info("Unprepare")
    k_recon = unpad_output(k_space=k_recon, padding=padding)
    torch.cuda.empty_cache()

    logger.info("Unbatch / Reshape")
    k_recon = unprepare_batches_to_k_space(
        k_batched=k_recon, batch_channel_indices=batch_channel_indices, original_shape=input_shape
    )
    return k_recon


def mc_loop(k: torch.Tensor, loraks: LoraksBase, noise_var: float = 1.0, num_iter: int = 50):
    k_mc = []
    for m in range(num_iter):
        logger.info(f"MC Run: {m + 1} / {num_iter}")
        # add noise to k
        noise = torch.randn_like(k)
        noise[k.abs() < 1e-9] = 0
        k_noise = k[:, :, k.shape[2] // 2, None] + noise_var * noise
        k_recon = loraks_recon(k_noise, loraks)
        k_mc.append(k_recon)
    return torch.stack(k_mc, dim=0)


def compute_stats_plot(img_mc: torch.Tensor, img_recon: torch.Tensor, path: plib.Path, name: str):
    # squeeze and move iteration dim to fron
    # img_mc = img_mc.squeeze().permute(2,0,1,3,4)

    # compute stats
    logger.info(f"Compute stats")
    mc_mean = torch.mean(img_mc, dim=0)
    mc_std = torch.std(img_mc, dim=0)
    plot_data(torch.stack([img_recon, mc_mean, img_recon - mc_mean], dim=2).squeeze(), path, f"{name}_mean")
    plot_data(torch.stack([img_recon, mc_mean, mc_std], dim=2).squeeze(), path, f"{name}_std")
    return mc_std, mc_mean


def mc_gfactor(k_noise: torch.Tensor, k_raw_slices: torch.Tensor, path: plib.Path, img_reference: torch.Tensor = None,
               device: torch.device = torch.get_default_device()):
    # we need to set up LORAKS recon
    logger.info(f"Setup LORAKS")
    opts = AcLoraksOptions(device=device)
    loraks = Loraks.create(opts)

    # get noise variance
    k_noise = k_noise.permute(1, 0, 2)
    k_noise = k_noise.reshape(k_noise.shape[0], -1)

    k_noise_var = k_noise.std(dim=-1)
    # as expected (with pre-whitening) the noise variance is 1.
    noise = torch.randn_like(k_noise)
    logger.info(f"Noise std: {noise.std().item():.2f}")

    # do unperturbed recon
    name_recon = "recon"
    if not path.joinpath(name_recon).with_suffix(".pt").exists():
        k_recon = loraks_recon(k_raw_slices, loraks)
        torch_save(k_recon, path, name_recon)
    else:
        k_recon = torch_load(path.joinpath(name_recon).with_suffix(".pt"))
    img_recon = fft_to_img(k_recon, dims=(0, 1))
    img_input = fft_to_img(k_raw_slices, dims=(0, 1))
    pd = [img_reference, img_input, img_recon] if img_reference is not None else [img_input, img_recon]
    pd = [d[:, :, d.shape[2] // 2] for d in pd]
    plot_data(
        torch.stack(pd, dim=2).squeeze(),
        path=path, name="input_recon"
    )
    # do mc perturbations with undersampled input
    name_iter = "mc_iterations_50"
    if not path.joinpath(name_iter).with_suffix(".pt").exists():
        k_mc_acc = mc_loop(k_raw_slices, loraks, noise_var=k_noise_var.mean().item())
        img_mc = fft_to_img(k_mc_acc, dims=(1, 2))
        torch_save(img_mc, path, name_iter)
    else:
        img_mc = torch_load(path.joinpath(name_iter).with_suffix(".pt"))
    # reduce load for now just use first echo
    img_tmp = img_mc[..., 0, None].clone()
    del img_mc
    img_mc = img_tmp
    del img_tmp
    torch.cuda.empty_cache()
    img_tmp = img_recon[..., 0, None].clone()
    del img_recon
    img_recon = img_tmp
    del img_tmp
    torch.cuda.empty_cache()

    # use denoised data if no reference is given
    if img_reference is None:
        name_iter = "k_denoised"
        if not path.joinpath(name_iter).with_suffix(".pt").exists():
            img_recon_denoised, _, _ = denoise_lcpca(img_recon, p=1, device=device)
            k_recon_denoised = ifft_to_k(img_recon_denoised, dims=(0, 1))
            torch_save(k_recon_denoised, path, name_iter)
        else:
            k_recon_denoised = torch_load(path.joinpath(name_iter).with_suffix(".pt"))
        img_reference = fft_to_img(k_recon_denoised, dims=(0, 1))

    img_tmp = img_reference[..., 0, None].clone()
    del img_reference
    img_reference = img_tmp
    del img_tmp
    torch.cuda.empty_cache()

    # otherwise use reference
    k_ref = ifft_to_k(img_reference, dims=(0, 1))
    name_iter = "mc_reference"
    if not path.joinpath(name_iter).with_suffix(".pt").exists():
        # do the simulation
        noise = torch.randn((50, *k_recon.shape), dtype=k_recon.dtype, device=k_recon.device)
        k_mc_ref_noise = k_ref[None] + noise
        img_mc_ref = fft_to_img(k_mc_ref_noise, dims=(1, 2))
        torch_save(img_mc_ref, path, name_iter)
    else:
        img_mc_ref = torch_load(path.joinpath(name_iter).with_suffix(".pt"))
    # reduce load for now just use first echo
    img_tmp = img_mc_ref[..., 0, None].clone()
    del img_mc_ref
    img_mc_ref = img_tmp
    del img_tmp
    torch.cuda.empty_cache()

    mc_std, mc_mean = compute_stats_plot(
        img_mc[:, :, :, img_mc.shape[3] // 2],
        img_recon[:, :, img_recon.shape[2] // 2],
        path, "03_mc_stats"
    )
    mc_ref_std, mc_ref_mean = compute_stats_plot(
        img_mc_ref[:, :, :, img_mc_ref.shape[3] // 2],
        img_recon[:, :, img_recon.shape[2] // 2],
        path, "03_mc_ref_stats"
    )

    # g-factor
    logger.info(f"Normalise and G-factor")
    g_factor = mc_std / mc_ref_std * np.sqrt(2)
    g_factor = smooth_map(data=g_factor, kernel_size=8)

    plot_data(
        torch.stack([img_recon[:, :, img_recon.shape[2] // 2], mc_mean, g_factor], dim=2).squeeze(),
        zmax=[None, None, 1.5], zmin=[None, None, 1],
        path=path, name="g-factor")


def set_paths(name, data: bool = False):
    path = plib.Path(
        get_test_result_output_dir(name, mode=ResultMode.EXPERIMENT)
    )
    if data:
        path_data = plib.Path(
            get_test_result_output_dir(name, mode=ResultMode.DATA)
        )

        return path, path_data
    return path


def g_factor_shepp_logan(device: torch.device = torch.get_default_device()):
    logger.info(f"MC simulation SheppLogan")
    path = set_paths(g_factor_shepp_logan)

    # built data
    phantom = Phantom.get_shepp_logan(shape=(160, 200), num_coils=4, num_echoes=2)
    k_ref = phantom.get_2d_k_space() * 1e4
    img_ref = fft_to_img(k_ref, dims=(0, 1))
    k_us = phantom.sub_sample_ac_random_lines(acceleration=3, ac_lines=32) * 1e4

    k_noise = torch.randn((10, 4, 1000), device=k_us.device, dtype=k_us.dtype)

    mc_gfactor(k_noise=k_noise, k_raw_slices=k_us.unsqueeze(2), path=path, img_reference=img_ref.unsqueeze(2), device=device)


def g_factor_phantom_data(device: torch.device = torch.get_default_device()):
    logger.info(f"MC simulation phantom data")
    path, path_data = set_paths(g_factor_phantom_data, data=True)

    # load in data
    logging.info("Load data")
    k_raw_slices = torch_load(path_data.joinpath("k_raw_slices").with_suffix(".pt"))
    img_in = fft_to_img(k_raw_slices, dims=(0, 1))
    plot_data(img_in, path, "01_input")

    # load noise
    k_noise = torch_load(path_data.joinpath("k_noise_scans").with_suffix(".pt"))

    mc_gfactor(k_noise=k_noise, k_raw_slices=k_raw_slices, path=path, device=device)


if __name__ == '__main__':
    setup_program_logging("LORAKS - G-factor", logging.INFO)
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"- device: {torch.cuda.get_device_name(device)}")
    g_factor_shepp_logan(device=device)
    g_factor_phantom_data(device=device)

