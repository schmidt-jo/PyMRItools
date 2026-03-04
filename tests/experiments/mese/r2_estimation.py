import torch
import numpy as np
import logging
import pathlib as plib
from scipy.stats import norm

import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.config import setup_program_logging
from pymritools.config.database import DB
from pymritools.utils import torch_load, torch_save, fft_to_img, ifft_to_k
from pymritools.modeling.espirit.functions import map_estimation
from pymritools.processing.denoising.stats import non_central_chi as ncc_stats
from pymritools.processing.denoising.lcpca import noise_bias_correction, denoise_lcpca
from pymritools.modeling.dictionary.grid_search_channels import fit_t2b1, normalise_data, get_normalise_database, \
    smooth_map, fit_regularised_t2

from tests.utils import get_test_result_output_dir, ResultMode

logger = logging.getLogger(__name__)


def get_noise_sigma_from_rayleigh(noise_data: torch.Tensor):
    if torch.is_complex(noise_data):
        noise_data = noise_data.abs()
    noise_sigma_m = np.sqrt(2 / np.pi) * noise_data.mean()
    noise_sigma_s = np.sqrt(2 / (4 - np.pi)) * noise_data.std()
    logging.info(f"Noise sigma through mean: {noise_sigma_m:.3f}, through std: {noise_sigma_s:.3f}")
    return noise_sigma_m



def espirit_estimate_sensitivities(path: plib.Path, img: torch.Tensor, k: torch.Tensor):
    if not path.joinpath("espirit_maps.pt").exists():
        logging.info("ESPIRIT style coil sensitivity estimation")
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
        logging.info(f"Write file: {fn}")
        fig.write_html(fn.as_posix()) if suff == ".html" else fig.write_image(fn.as_posix())


def main():
    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"- device: {torch.cuda.get_device_name(device)}")

    path = plib.Path(
        get_test_result_output_dir("r2_estimation_cfa".lower(), mode=ResultMode.EXPERIMENT
        )
    )
    path_data = plib.Path(
        get_test_result_output_dir("r2_phantom", mode=ResultMode.DATA)
    )
    # load in data
    logging.info("Load data")
    # k_recon = torch_load(path_data.joinpath("mese_cfa_rf2838_ga_rgs0p8_a3p65_trd20_m095sl82g67_denoised").with_suffix(".pt"))
    # torch_save(k_recon[:, :, k_recon.shape[2] // 2].clone(), path, file_name="k_recon_slice")
    img_recon = torch_load(path_data.joinpath("k_recon_slice").with_suffix(".pt")).unsqueeze(2)
    k_recon = ifft_to_k(img_recon, dims=(0, 1))

    # we estimate the noise distribution from the corners (complex data)
    mask_noise = torch.zeros(img_recon.shape, dtype=torch.bool)
    edge_size = 35
    mask_noise[:edge_size, :edge_size] = True
    mask_noise[:edge_size, -edge_size:] = True
    mask_noise[-edge_size:, :edge_size] = True
    mask_noise[-edge_size:, -edge_size:] = True

    noise_data = img_recon[mask_noise]

    # extract noise sigma
    logging.info("Extract noise sigma")
    noise_sigma = get_noise_sigma_from_rayleigh(noise_data)
    plot_noise_stats(
        noise_data=noise_data, noise_sigma=noise_sigma, path=path, suffix=[".html", ".pdf"]
    )

    # assume nsig**2 to be constant
    img_nbc = noise_bias_correction(img_recon, sigma=noise_sigma, num_channels=1)

    plot_maps([img_recon, img_nbc, torch.abs(img_recon) - img_nbc], path=path, name="noise_bias_correction",
              suffix=[".html", ".pdf"],
              zs=[(0, img_recon.abs().max().item()*0.2)] * 2 + [(0, img_recon.abs().max().item()*0.01)])

    logging.info("Start estimation ESPIRiT style sensitivity maps")
    # estimate sensitivity maps
    espirit_maps = espirit_estimate_sensitivities(path, img_nbc, k_recon)
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

    # normalise dataa
    data, data_norm = normalise_data(img_nbc)
    # batch data
    data_shape = data.shape

    path_t2 = path.joinpath("t2.pt")
    if not path_t2.exists():
        # estimate parameters
        logging.info("Start estimation of T2 and B1")
        # fit t2 and b1
        t2, b1, _ = fit_t2b1(
            data_normed_rpsct=data, db_normed=db_normed,
            db=db, device=device
        )
        torch_save(t2, path, file_name="t2")
        torch_save(b1, path, file_name="b1")
    else:
        t2 = torch_load(path_t2)
        b1 = torch_load(path.joinpath("b1.pt"))

    plot_maps([img_nbc[..., 0], t2*1e3, b1], path=path, name="fit_maps")

    t2_wavg = torch.sum(smaps * t2, dim=-1) / smaps_sum * 1e3
    b1_wavg = torch.sum(smaps * b1, dim=-1) / smaps_sum
    img_wavt = torch.sum(smaps * img_nbc[..., 0], dim=-1) / smaps_sum

    # denoise
    path_t2 = path.joinpath("t2_wavg_den.pt")
    if not path_t2.exists():
        t2_wavg_den, _, _ = denoise_lcpca(
            input_data=t2_wavg.unsqueeze(-1).unsqueeze(-1).expand(*k_recon.shape[:3], 1, 8), p=1, device=device,
        )
        torch_save(t2_wavg_den[..., 0], path, file_name="t2_wavg_den")
    else:
        t2_wavg_den = torch_load(path_t2)

    # plot maps
    logging.info("Plot maps")
    plot_maps(maps=[img_wavt, t2_wavg, t2_wavg_den, b1_wavg], path=path, name="fit_wavt",
              zs=[(None, None), (0, 150), (0, 150), (0.5, 1.5)])

    if not path.joinpath("b1_smoothed.pt").exists():
        logging.info(f"smooth B1")
        b1_map = smooth_map(b1, kernel_size=min(b1.shape[:2]) // 20)
        torch_save(b1_map, path, file_name="b1_smoothed")

        # fit regularised
        logging.info(f"fit with regularisation, input smoothed B1")
        t2_reg, l2_res = fit_regularised_t2(
            data_normed_rpsct=data, b1_map=b1_map,
            db_normed=db_normed, db=db, device=device
        )
        torch_save(t2_reg, path, file_name="t2_reg")

    else:
        b1_map = torch_load(path.joinpath("b1_smoothed.pt"))
        t2_reg = torch_load(path.joinpath("t2_reg.pt"))

    plot_maps([img_nbc[..., 0], t2_reg*1e3, b1_map], path=path, name="fit_maps_reg")

    # weighted average combination
    smaps = sensitivity_maps.abs()
    smaps_sum = torch.sum(smaps.abs(), dim=-1)
    t2_wavg = torch.sum(smaps * t2_reg, dim=-1) / smaps_sum * 1e3
    b1_wavg = torch.sum(smaps * b1_map, dim=-1) / smaps_sum
    img_wavt = torch.sum(smaps * img_nbc[..., 0], dim=-1) / smaps_sum


    # plot maps
    logging.info("Plot maps")
    plot_maps(maps=[img_wavt, t2_wavg, b1_wavg], path=path, name="fit_reg_wavt")


    # redo with rsos
    data_rsos = torch.sqrt(torch.sum(torch.square(data.abs()), dim=-2))
    data_rsos_normed, data_rsos_norm = normalise_data(data_rsos)
    data_rsos_normed = data_rsos_normed.unsqueeze(-2)

    path_t2 = path.joinpath("t2_rsos.pt")
    if not path_t2.exists():
        # estimate parameters
        logging.info("Start estimation of T2 and B1")
        # fit t2 and b1
        t2_rsos, b1_rsos, _ = fit_t2b1(
            data_normed_rpsct=data_rsos_normed, db_normed=db_normed,
            db=db, device=device
        )
        t2_rsos = t2_rsos[..., 0]
        b1_rsos = b1_rsos[..., 0]
        torch_save(t2_rsos, path, file_name="t2_rsos")
        torch_save(b1_rsos, path, file_name="b1_rsos")
    else:
        t2_rsos = torch_load(path_t2)
        b1_rsos = torch_load(path.joinpath("b1_rsos.pt"))

    plot_maps([data_rsos[..., 0], t2_rsos*1e3, b1_rsos], path=path, name="fit_maps_rsos")

def plot_maps(maps: list,
              path: plib.Path, name:str, zs: list = None, suffix: str | list = ".html"):
    if zs is None:
        zs = [(None, None), (0, 120), (0.5, 1.5)]
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
                        z=d.abs(), transpose=True, colorscale="Inferno", showscale=False,
                        zmin=zs[j][0], zmax=zs[j][1]
                    ),
                    row=1 + j, col=1 + i
                )
        else:
            fig.add_trace(
                go.Heatmap(
                    z=dn[:, :, 0].abs(), transpose=True, colorscale="Inferno", showscale=False,
                    zmin=zs[j][0], zmax=zs[j][1]
                ),
                row=1, col=1 + j
            )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    if not isinstance(suffix, list):
        suffix = [suffix]
    for suff in suffix:
        fn = path.joinpath(name).with_suffix(suff)
        logging.info(f"Write file: {fn}")
        fig.write_html(fn.as_posix()) if suff == ".html" else fig.write_image(fn.as_posix())


if __name__ == '__main__':
    setup_program_logging("R2 estimation", logging.INFO)
    main()
