"""
This script is intended to optimize the mese (or megesse) pulse trains to produce dictionary curves with
minimal SAR while offering maximal SNR.
"""
import sys
import logging
import pathlib as plib
import pickle
import json

import polars as pl
import torch
import tqdm
from scipy.constants import physical_constants
import plotly.graph_objects as go
import plotly.colors as plc

from pymritools.simulation.emc.sequence.mese import MESE

from pymritools.config.emc import EmcSimSettings, EmcParameters, SimulationData
from pymritools.simulation.emc.core.blocks import GradPulse
from pymritools.simulation.emc.core.functions import (
    matrix_propagation_relaxation_multidim, propagate_gradient_pulse_relax,
    propagate_matrix_mag_vector, sum_sample_acquisition
)
path = plib.Path(__name__).absolute().parent
sys.path.append(path.as_posix())

from tests.utils import get_test_result_output_dir, ResultMode


def main():
    # hardcode some of the path and parameters
    path = plib.Path(__name__).absolute().parent
    path_out = plib.Path(
        get_test_result_output_dir("vfa_mese_autograd", mode=ResultMode.OPTIMIZATION)
    ).joinpath("optim_run")
    path_out.mkdir(exist_ok=True, parents=True)
    # setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO, filename=path_out.joinpath("log.txt").as_posix(),
        filemode="w"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings_path = path.joinpath("optimization/optim_emc_settings.json")
    settings = EmcSimSettings.load(settings_path.as_posix())
    settings.visualize = False

    # we need to fix some path business in the example file
    for key, val in settings.__dict__.items():
        if key.endswith("file"):
            p = path.joinpath(val)
            settings.__setattr__(key, p.as_posix())

    settings.display()
    # testing: set to simulate low number of vals
    # settings.t2_list = [35, 45, 55, 65]
    # settings.b1_list = [0.5, 1.0]
    params = EmcParameters.load(settings.emc_params_file)
    mese = MESE(settings=settings, params=params)
    pulse_x = [mese.gps_refocus[i].data_pulse_x for i in range(len(mese.gps_refocus))]
    pulse_y = [mese.gps_refocus[i].data_pulse_y for i in range(len(mese.gps_refocus))]
    # setup fas
    lam_snr = 0.5
    lam_corr = 1e-4

    fa_ref = torch.full((params.etl,), fill_value=0.8, device="cpu", requires_grad=True)
    fa_0 = torch.tensor(params.refocus_angle)

    losses = []
    losses_snr = []
    losses_sar = []
    losses_corr = []
    fas = []

    max_num_iter = 50
    lr = torch.linspace(5e-1, 1e-2, max_num_iter)

    conv_count = 0
    bar = tqdm.trange(max_num_iter)
    for idx in bar:
        # easy sar loss
        sar = torch.sum(torch.abs(fa_ref))
        # set up data, carrying through simulation
        mese.data = SimulationData(params=mese.params, settings=mese.settings, device=mese.device)

        for i in range(fa_ref.shape[0]):
            mese.gps_refocus[i].data_pulse_x = pulse_x[i] * fa_ref[i]
            mese.gps_refocus[i].data_pulse_y = pulse_y[i] * fa_ref[i]
        mese.simulate(use_prog_bar=False)
        # compute losses
        # maximize signal
        snr = - torch.linalg.norm(mese.data.signal_mag, dim=-1).flatten().mean() * 10
        # minimize correlations between r2 curves
        curves = torch.squeeze(mese.data.signal_mag)
        curves = curves / torch.linalg.norm(curves, dim=-1, keepdim=True)
        curves = curves.view(curves.shape[0], -1)
        num_c = curves.shape[-1]
        cov = torch.cov(curves) / num_c
        corr = torch.sum(cov)
        # corr = torch.zeros(1)
        # minimize sar, maximize snr, with a minimizing total loss

        loss = lam_snr * snr + lam_corr * corr + (1.0 - lam_snr - lam_corr) * sar
        loss.backward()
        with torch.no_grad():
            fa_ref -= lr[idx] * fa_ref.grad

        convergence = abs(loss.item() - losses[-1]) if idx > 0 else 1e3

        pr = {
            "loss": f"{loss.item():.4f}", "sar": f"{sar.item():.4f}", "snr": f"{snr.item():.4f}",
            "corr": f"{corr.item():.4f}", "convergence": f"{convergence:.4f}"
        }
        bar.postfix = pr
        losses.append(loss.item())
        losses_snr.append(snr.item())
        losses_sar.append(sar.item())
        losses_corr.append(corr.item())
        if convergence < 1e-5:
            conv_count += 1
            if conv_count > 3:
                logging.info(f"reached convergence at iteration {idx}")
                break
        fa = fa_ref.clone().detach().cpu() * fa_0
        fas.append(fa.tolist())
        fa_ref.grad.zero_()

    df_losses = pl.DataFrame({
        "loss": losses, "sar": losses_snr, "snr": losses_sar, "run": torch.arange(idx + 1).tolist(), "corr": losses_corr
    })
    fas = torch.tensor(fas)

    fig = go.Figure()
    for i, idf in enumerate(["loss", "snr", "sar", "corr"]):
        fig.add_trace(
            go.Scattergl(
                y=df_losses[idf], name=idf
            )
        )
    fn = path_out.joinpath("optimization_losses").with_suffix(".html")
    logging.info(f"Write file: {fn}")
    fig.write_html(fn.as_posix())

    fn = path_out.joinpath("losses_df").with_suffix(".json")
    logging.info(f"Write file: {fn}")
    df_losses.write_json(fn.as_posix())

    cmap = plc.sample_colorscale("Inferno", fas.shape[0], 0.1, 0.9)
    fig = go.Figure()
    for i, fa in enumerate(fas):
        fig.add_trace(
            go.Scatter(y=fa, showlegend=False, marker=dict(color=cmap[i]))
        )
    fn = path_out.joinpath("fa_optimization").with_suffix(".html")
    logging.info(f"Write file: {fn}")
    fig.write_html(fn.as_posix())

    opt = {
        "fas": fas[-1].tolist(),
        "lam_snr": lam_snr,
        "lam_corr": lam_corr,
        "lam_sar": (1 - lam_corr - lam_snr)
    }
    fn = path_out.joinpath("fa_optimized").with_suffix(".json")
    logging.info(f"Write file: {fn}")
    with open(fn.as_posix(), "w") as j_file:
        json.dump(opt, j_file, indent=2)


if __name__ == '__main__':
    main()
