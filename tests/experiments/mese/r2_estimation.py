import torch
import numpy as np
import logging
import pathlib as plib
from scipy.stats import norm
from nibabel.processing import resample_from_to
import nibabel as nib
import scipy.ndimage as ndi
from skimage.registration import phase_cross_correlation

import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.config import setup_program_logging
from pymritools.config.database import DB
from pymritools.utils import torch_load, torch_save, fft_to_img, ifft_to_k, adaptive_combine, nifti_load, \
    root_sum_of_squares
from pymritools.modeling.espirit.functions import map_estimation
from pymritools.processing.denoising.stats import non_central_chi as ncc_stats
from pymritools.processing.denoising.lcpca import noise_bias_correction, denoise_lcpca
from pymritools.modeling.dictionary.grid_search_channels import fit_t2b1, normalise_data, get_normalise_database, \
    smooth_map, fit_regularised_t2

from tests.utils import get_test_result_output_dir, ResultMode
from pymritools.utils import colormaps

logger = logging.getLogger(__name__)


def get_noise_sigma_from_rayleigh(noise_data: torch.Tensor):
    if torch.is_complex(noise_data):
        noise_data = noise_data.abs()
    noise_sigma_m = np.sqrt(2 / np.pi) * noise_data.mean()
    noise_sigma_s = np.sqrt(2 / (4 - np.pi)) * noise_data.std()
    logger.info(f"Noise sigma through mean: {noise_sigma_m:.3f}, through std: {noise_sigma_s:.3f}")
    return noise_sigma_m



def espirit_estimate_sensitivities(path: plib.Path, img: torch.Tensor, k: torch.Tensor):
    if not path.joinpath("espirit_maps.pt").exists():
        logger.info("ESPIRIT style coil sensitivity estimation")
        # espirit_maps = espirit_pytorch(
        #     k_rpsct=k.unsqueeze(2), kernel_size=6, n_ac=28, rank_fraction=0.01, ev_crop_thr=0.995,
        # )
        espirit_maps = map_estimation(
            k_rpsc=k, kernel_size=6, num_ac_lines=26, rank_fraction_ac_matrix=0.01, eigenvalue_cutoff=0.99
        )
        espirit_maps = torch.flip(espirit_maps, dims=(1, 2, 3))
        torch_save(espirit_maps, path, file_name="espirit_maps.pt")
    else:
        espirit_maps = torch_load(path.joinpath("espirit_maps.pt"))

    plot_maps([img, espirit_maps[0].unsqueeze(-1)], path=path, name="sensitivities", zs=[(None, None), (None, None)])
    return espirit_maps


def plot_noise_stats(noise_data: torch.Tensor, noise_sigma: float, path: plib.Path, suffix: str | list = ".html"):

    fig = psub.make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.02, horizontal_spacing=0.02

    )
    for j, f in enumerate([torch.view_as_real, torch.abs]):
        hist, bins = torch.histogram(f(noise_data).flatten(), bins=200)
        hist = hist / hist.sum()
        x = bins[:-1] + (bins[1] - bins[0]) / 2

        fig.add_trace(
            go.Bar(
                x=x, y=hist,
                marker=dict(color=["purple", "salmon"][j])
            ),
            row=1+j, col=1
        )
        if j == 0:
            y = norm(0, noise_sigma).pdf(x)
        else:
            # y = rayleigh(0, n_sig).pdf(x)
            y = ncc_stats.noise_dist_ncc(x, noise_sigma, 1)
        y = y / y.sum()
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                name=f"estimate",
                mode="lines", line=dict(color="teal")),
            row=1+j, col=1
        )
    if not isinstance(suffix, list):
        suffix = [suffix]
    for suff in suffix:
        fn = path.joinpath("noise_hist").with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_html(fn.as_posix()) if suff == ".html" else fig.write_image(fn.as_posix())


def main():
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"- device: {torch.cuda.get_device_name(device)}")

    path = plib.Path(
        get_test_result_output_dir("r2_estimation_vfa_1p05_ga_ew".lower(), mode=ResultMode.EXPERIMENT
        )
    )
    path_data = plib.Path(
        get_test_result_output_dir("r2_phantom", mode=ResultMode.DATA)
    )
    # load in data
    logger.info("Load data")
    img_recon = torch_load(path_data.joinpath("2025-08-27_mese_vfa_a3p75_sp3000_tr4p5_sl40_g65_rgs1p05_denoising_p1").with_suffix(".pt"))
    torch_save(img_recon[:, :, img_recon.shape[2] // 2].clone(), path, file_name="img_recon_slice")
    # load in afi
    _, b1_afi_img = nifti_load(
        path_data.joinpath("2025-12-03_afi_b1_hires").with_suffix(".nii")
    )
    _, b1_afi_ref_img = nifti_load(
        path_data.joinpath("2025-12-03_afi_ref_hires").with_suffix(".nii")
    )
    # load in niftii
    # den_nii, den_nii_img =nifti_load(
    #     path_data.joinpath("mese_vfa_rf2838_ga_rgs0p8_a3p65_trd20_m095sl82g67_re_t720_gem10_denoised_data").with_suffix(".nii")
    # )
    img_re_rsos = root_sum_of_squares(img_recon[..., 0], dim_channel=-1)
    den_nii_img = nib.Nifti1Image(img_re_rsos.numpy(), affine=np.eye(4))

    b1_afi_re_img_ref = resample_from_to(b1_afi_ref_img, den_nii_img, order=1, mode="nearest")
    b1_afi_re_img_map = resample_from_to(b1_afi_img, den_nii_img, order=1, mode="nearest")

    # register
    b1_afi_re_ref_data = b1_afi_re_img_ref.get_fdata()
    b1_afi_re_map_data = b1_afi_re_img_map.get_fdata()

    logger.info("Register afi")
    shift, error, diffphase = phase_cross_correlation(img_re_rsos, b1_afi_re_ref_data)
    afi_aligned_resampled_ref = ndi.shift(
        input=b1_afi_re_ref_data, shift=shift, order=1, mode="constant", cval=0
    )
    afi_aligned_resampled_map = ndi.shift(
        input=b1_afi_re_map_data, shift=shift, order=1, mode="constant", cval=0
    )

    torch.cuda.empty_cache()
    del b1_afi_re_img_ref, b1_afi_re_img_map, b1_afi_re_ref_data, b1_afi_re_map_data, img_re_rsos, den_nii_img, b1_afi_img, b1_afi_ref_img
    # for now just work with middle slice
    slice_idx = img_recon.shape[2] // 2 - 15
    img_recon = img_recon[:, :, slice_idx, None].clone()
    k_recon = ifft_to_k(img_recon, dims=(0, 1))
    logger.info(f"img_recon data shape: {img_recon.shape}")

    # we estimate the noise distribution from the corners (complex data)
    mask_noise = torch.zeros(img_recon.shape, dtype=torch.bool)
    edge_size = 35
    mask_noise[:edge_size, :edge_size] = True
    mask_noise[:edge_size, -edge_size:] = True
    mask_noise[-edge_size:, :edge_size] = True
    mask_noise[-edge_size:, -edge_size:] = True

    noise_data = img_recon[mask_noise]

    # extract noise sigma
    logger.info("Extract noise sigma")
    noise_sigma = get_noise_sigma_from_rayleigh(noise_data)
    plot_noise_stats(
        noise_data=noise_data, noise_sigma=noise_sigma, path=path, suffix=[".html", ".pdf"]
    )

    # assume nsig**2 to be constant
    img_nbc = noise_bias_correction(img_recon, sigma=noise_sigma, num_channels=1)

    plot_maps([img_recon, img_nbc, torch.abs(img_recon) - img_nbc],
              path=path, name="noise_bias_correction",
              suffix=[".html", ".pdf"],
              zs=[(0, img_recon.abs().max().item()*0.2)] * 2 + [(0, img_recon.abs().max().item()*0.01)])

    logger.info("Start estimation ESPIRiT style sensitivity maps")
    # estimate sensitivity maps
    espirit_maps = espirit_estimate_sensitivities(path, img_recon, k_recon)
    sensitivity_maps = espirit_maps[0]
    sensitivity_maps[sensitivity_maps.abs() < 1e-6] = 1

    # weighted average combination
    smaps = sensitivity_maps.abs()
    smaps_sum = torch.sum(smaps.abs(), dim=-1)

    plot_maps(
        [img_nbc[..., 0], sensitivity_maps],
        zs=[(0, img_nbc.abs().max().item()*0.2), (0, 1)], path=path, name="sensitivities", suffix=[".html", ".pdf"]
    )
    # load database
    path_database = plib.Path(
        "C:\\DatenJo\\Daten\\02_work\\04_data\\proj_mese\\emc\\databases\\"
        "db_mese_vfa-rf26-38-gauss_rgs0p8_a3_trd15_m065-sl82-g67_read-sp96-t720-ge-m100.pkl"
    )
    db = DB.load(path_database)
    # process and normalise
    db_normed, db_norm, db_shape = get_normalise_database(db, dtype=img_recon.dtype, device=device)

    # normalise data
    data, data_norm = normalise_data(img_nbc)
    # batch data
    data_shape = data.shape

    path_t2 = path.joinpath("t2.pt")
    if not path_t2.exists():
        # estimate parameters
        logger.info("Start estimation of T2 and B1")
        # fit t2 and b1
        t2, b1, _ = fit_t2b1(
            data_normed_rpsct=data, db_normed=db_normed,
            db=db, device=device, echo_weighting=True
        )
        torch_save(t2, path, file_name="t2")
        torch_save(b1, path, file_name="b1")
    else:
        t2 = torch_load(path_t2)
        b1 = torch_load(path.joinpath("b1.pt"))
    t2[t2 > 0.2] = 0.001
    plot_maps(
        maps=[img_nbc[..., 0], t2*1e3, b1],
        path=path, name="01_fit_maps",
        zs=[(None, None), (0, 150), (0.5, 1.5)],
        cmaps=["Inferno", colormaps.get_colormap("navia"), "Inferno"]
    )

    t2_wavg = torch.sum(smaps * t2, dim=-1) / smaps_sum * 1e3
    b1_wavg = torch.sum(smaps * b1, dim=-1) / smaps_sum
    b1_wavg = smooth_map(b1_wavg, kernel_size=min(b1_wavg.shape[:2]) // 30)
    img_wavt = torch.sum(smaps * img_nbc[..., 0], dim=-1) / smaps_sum

    # plot maps
    logger.info("Plot maps")
    plot_maps(
        maps=[img_wavt, torch.nan_to_num(1000/t2_wavg, nan=0.0, posinf=0.0), b1_wavg],
        path=path, name="02_fit_maps_wavt",
        zs=[(None, None), (0, 120), (0.5, 1.5)],
        cmaps=["Gray", colormaps.get_colormap("navia"), "Inferno"]
    )

    if not path.joinpath("b1_smoothed.pt").exists():
        logger.info(f"smooth B1")
        b1_map = smooth_map(b1, kernel_size=min(b1.shape[:2]) // 20)
        torch_save(b1_map, path, file_name="b1_smoothed")
    else:
        b1_map = torch_load(path.joinpath("b1_smoothed.pt"))

    logger.info(f"Combined B1")

    # combined afi fitting
    afi_slice = torch.from_numpy(afi_aligned_resampled_map[:, :, slice_idx, None])

    # for lambda_afi in np.linspace(0.1, 0.9, 5):
    lambda_afi = 0.0
    path_sub = path.joinpath(f"lafi-{lambda_afi:.2f}".replace(".", "p"))
    path_sub.mkdir(parents=True, exist_ok=True)

    # scale afi
    afi_min = afi_slice[afi_slice > 10].min()
    afi_max = afi_slice[afi_slice > 10].max()
    an_max = 150
    an_min = 70
    afi_slice_mapped = (afi_slice - afi_min) / (afi_max - afi_min) * (an_max - an_min) + an_min
    b1_comb = (1 - lambda_afi) * b1_wavg + lambda_afi * afi_slice / 100
    b1_comb[afi_slice < 1e-6] = b1_wavg[afi_slice < 1e-6].to(torch.float64)
    plot_maps([b1_wavg, afi_slice, afi_slice_mapped, b1_comb], path=path_sub,
              name=f"05_fit_afi_comb",
              zs=[(0.5, 1.5), (50, 150), (50, 150), (0.3, 1.2)],
              cmaps=["Inferno", "Inferno", "Inferno", "Inferno"])
    b1_map = b1_comb

    if not path_sub.joinpath("t2_reg.pt").exists():
        # fit regularised
        logger.info(f"fit with regularisation, input smoothed B1")
        t2_reg, l2_res = fit_regularised_t2(
            data_normed_rpsct=data, b1_map=b1_map,
            db_normed=db_normed, db=db, device=device, echo_weighting=True
        )
        torch_save(t2_reg, path_sub, file_name="t2_reg")
    else:
        t2_reg = torch_load(path_sub.joinpath("t2_reg.pt"))
    t2_reg[t2_reg > 0.2] = 0.001
    # smooth / denoise t2 maps
    t2_reg_den, _, _ = denoise_lcpca(
        input_data=t2_reg.unsqueeze(-1), p=1, fixed_cube_side_length=4
    )
    t2_reg_den = t2_reg_den.unsqueeze(-2).expand_as(t2_reg)
    b1_map = b1_map.unsqueeze(-1).expand_as(t2_reg)
    plot_maps(
        maps=[img_nbc[..., 0], t2_reg*1e3,  t2_reg_den*1e3, sensitivity_maps.abs(), b1_map],
        path=path_sub, name="03_fit_maps_reg",
        zs=[(None, None), (0, 100), (0, 100), (0, 1.2), (0.3, 1.2)],
        cmaps=["Inferno", colormaps.get_colormap("navia"), colormaps.get_colormap("navia"), "Inferno", "Inferno"]
    )
    t2_reg[t2_reg < 1e-3] = 1e-3
    t2_reg_den[t2_reg_den < 1e-3] = 1e-3

    r2_reg = torch.nan_to_num(1/t2_reg, nan=0.0, posinf=0.0)
    r2_reg_den = torch.nan_to_num(1/t2_reg_den, nan=0.0, posinf=0.0)

    r2_reg[r2_reg > 250] = 0
    r2_reg_den[r2_reg_den > 250] = 0

    r2_reg_den, _, _ = denoise_lcpca(
        input_data=r2_reg.unsqueeze(-1), p=1, fixed_cube_side_length=3
    )
    r2_reg_den = r2_reg_den.unsqueeze(-2).expand_as(r2_reg)
    # weighted average combination
    smaps = sensitivity_maps.abs()
    smaps[smaps < 1e-2] = 0
    smaps_sum = torch.nan_to_num(1 / torch.sum(smaps.abs(), dim=-1), nan=0.0, posinf=0.0)
    r2_wavg = torch.sum(smaps * r2_reg, dim=-1) * smaps_sum
    r2_wavg_den = torch.sum(smaps * r2_reg_den, dim=-1) * smaps_sum
    b1_wavg = torch.sum(smaps * b1_map, dim=-1) * smaps_sum
    img_wavt = torch.sum(smaps * img_nbc[..., 0], dim=-1) * smaps_sum

    torch_save(r2_wavg, path_sub, file_name="r2_wavg")
    torch_save(r2_wavg_den, path_sub, file_name="r2_wavg_den")
    # plot maps
    logger.info("Plot maps")
    plot_maps(
        maps=[img_wavt, r2_wavg, r2_wavg_den, b1_wavg],
        path=path_sub, name="04_fit_reg_wavt",
        zs=[(None, None), (0, 120), (0, 120), (0.3, 1.2)],
        cmaps = ["Inferno", colormaps.get_colormap("navia"), colormaps.get_colormap("navia"), "Inferno"]
    )

    # redo with rsos
    data_rsos = torch.sqrt(torch.sum(torch.square(data.abs()), dim=-2))
    data_rsos_normed, data_rsos_norm = normalise_data(data_rsos)
    data_rsos_normed = data_rsos_normed.unsqueeze(-2)

    path_t2 = path.joinpath("t2_rsos.pt")
    if not path_t2.exists():
        # estimate parameters
        logger.info("Start estimation of T2 and B1")
        # fit t2 and b1
        t2_rsos, l2_res_rsos = fit_regularised_t2(
            data_normed_rpsct=data_rsos_normed, b1_map=b1_comb, db_normed=db_normed,
            db=db, device=device
        )
        t2_rsos = t2_rsos[..., 0]
        # b1_rsos = b1_rsos[..., 0]
        b1_rsos = b1_comb
        torch_save(t2_rsos, path, file_name="t2_rsos")
        # torch_save(b1_rsos, path, file_name="b1_rsos")
    else:
        t2_rsos = torch_load(path_t2)
        b1_rsos = torch_load(path.joinpath("b1_rsos.pt"))

    plot_maps(
        maps=[data_rsos[..., 0], torch.nan_to_num(1/t2_rsos, nan=0.0, posinf=0.0), b1_rsos],
        path=path, name="05_fit_maps_rsos",
        zs=[(None, None), (0, 50), (0.3, 1.2)],
        cmaps=["Inferno", colormaps.get_colormap("navia"), "Inferno"]
    )


def extract_values():
    path = plib.Path(
        get_test_result_output_dir("r2_estimation_vfa_ga_nbc".lower(), mode=ResultMode.EXPERIMENT
                                   )
    )
    path_data = path.joinpath("lafi-0p75")
    path_out = path.joinpath("values")
    path_out.mkdir(parents=True, exist_ok=True)
    # load in data
    logger.info("Load data")
    t2_wavg = torch_load(path_data.joinpath("t2_wavg").with_suffix(".pt"))
    r2 = torch.nan_to_num(1000/t2_wavg, nan=0.0, posinf=0.0)

    # build vial mask
    vial_mask = torch.zeros_like(r2)
    concenter = torch.tensor([s // 2 for s in r2.shape], dtype=torch.long)

    vial_mask[concenter[0]-3:concenter[0]+3, concenter[1]-3:concenter[1]+3] = 1

    plot_maps(
        maps=[r2, vial_mask],
        zs=[(0, 120), (0, 1)],
        path=path_out, name="01_r2_vial_mask",
    )


def plot_maps(maps: list,
              path: plib.Path, name:str,
              zs: list = None,
              cmaps: list = None,
              suffix: str | list = ".html"):
    if zs is None:
        zs = [(None, None), (0, 120), (0.5, 1.5)]
    if cmaps is None:
        cmaps = ["Inferno"] * len(maps)
    if maps[0].ndim > 3:
        fig = psub.make_subplots(rows=len(maps), cols=5)
    else:
        fig = psub.make_subplots(rows=1, cols=len(maps))
    for j, dn in enumerate(maps):
        if maps[0].ndim > 4:
            dn = dn[..., 0]
        if maps[0].ndim > 3:
            for i, d in enumerate(dn[:, :, 0, 10:15].permute(2, 0, 1)):
                fig.add_trace(
                    go.Heatmap(
                        z=d.abs(), transpose=True, colorscale=cmaps[j], showscale=False,
                        zmin=zs[j][0], zmax=zs[j][1]
                    ),
                    row=1 + j, col=1 + i
                )
        else:
            fig.add_trace(
                go.Heatmap(
                    z=dn[:, :, 0].abs(), transpose=True, colorscale=cmaps[j], showscale=False,
                    zmin=zs[j][0], zmax=zs[j][1],
                ),
                row=1, col=1 + j
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if not isinstance(suffix, list):
        suffix = [suffix]
    for suff in suffix:
        fn = path.joinpath(name).with_suffix(suff)
        logger.info(f"Write file: {fn}")
        fig.write_html(fn.as_posix()) if suff == ".html" else fig.write_image(fn.as_posix())


if __name__ == '__main__':
    setup_program_logging("R2 estimation", logging.INFO)
    main()
    extract_values()
