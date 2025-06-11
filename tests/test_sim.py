import logging

import torch
from tests.utils import get_test_result_output_dir, ResultMode
from pymritools.config.emc.settings import SimulationSettings
from pymritools.config.emc.params import Parameters
from pymritools.simulation.emc.sequence.mese import MESE
import pathlib as plib

import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc


def sim_run(settings: SimulationSettings, params: Parameters, name: str, path_out: plib.Path):
    # re-direct output
    settings.out_path = path_out.joinpath(name).as_posix()

    # set up sim
    seq_sim = MESE(params=params, settings=settings)
    # simulate
    seq_sim.simulate()
    seq_sim.save()
    mag_profiles = seq_sim._fig_magnetization_profile_snaps
    plot_profiles(
        mag_profiles=mag_profiles, sample_axis=seq_sim.data.sample_axis,
        path_out=path_out.joinpath(f"{name}/figs"), name="mag_profiles"
    )


def test_sim_emc():
    # get output dir
    path_base = plib.Path(__name__).absolute().parent.parent

    path_out = plib.Path(
        get_test_result_output_dir(func="sim_emc", mode=ResultMode.TEST)
    ).absolute()

    logging.basicConfig(
        filename=path_out.joinpath('test.log').as_posix(), encoding='utf-8', level=logging.INFO,
        format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', filemode="w"
    )

    path_in = path_base.joinpath("examples/simulation/emc/emc_settings.json")
    # load in example config to test a standard MESE protocol
    settings = SimulationSettings.load(path_in)

    # we need to fix some path business in the example file
    for key, val in settings.__dict__.items():
        if key.endswith("file"):
            p = path_base.joinpath(val)
            settings.__setattr__(key, p.as_posix())
    # want to test the simulation for a handful of t2s and 2 b1s
    settings.display()

    logging.info(f"__Simulate 180 refocusing__\n")
    params = Parameters.load(settings.emc_params_file)
    params.refocus_angle = [180.0] * params.etl
    sim_run(settings=settings, params=params, name="run_results_180", path_out=path_out)

    logging.info(f"__Simulate 90 refocusing__\n")
    params.refocus_angle = [90.0] * params.etl
    sim_run(settings=settings, params=params, name="run_results_90", path_out=path_out)

    logging.info(f"__Simulate cfa 140 refocusing__\n")
    params.refocus_angle = [140.0] * params.etl
    sim_run(settings=settings, params=params, name="run_results_cfa", path_out=path_out)

    logging.info(f"__Simulate vfa refocusing__\n")
    params.refocus_angle = [136.6, 133.2, 110.8, 101.5, 91.1, 82.4, 87.9, 95.8]
    sim_run(settings=settings, params=params, name="run_results_vfa", path_out=path_out)

    logging.info(f"__Simulate 90 degree pulse__\n")
    params = Parameters.load(settings.emc_params_file)
    params.etl = 1
    params.excitation_angle = 90.0
    params.refocus_angle = [0.0]
    params.refocus_phase = [0.0]
    sim_run(settings=settings, params=params, name="run_results_pulse", path_out=path_out)


def plot_profiles(mag_profiles: list, sample_axis: torch.Tensor, path_out: plib.Path, name: str):
    mag_profiles = [m for m in mag_profiles if m["name"].startswith("initial") or m["name"].startswith("excitation") or m["name"].endswith("pulse")]
    fig = psub.make_subplots(
        rows=len(mag_profiles), cols=1,
        row_titles=[p["name"] for p in mag_profiles],
    )
    cmap = plc.sample_colorscale("Inferno", 3, 0.2, 0.9)
    for i, m in enumerate(mag_profiles):
        m = m["profile"]
        for k, p in enumerate([m[:, -1], m[:, -2], torch.linalg.norm(m[:, :2], dim=-1)]):
            fig.add_trace(
                go.Scatter(
                    y=p, x=sample_axis.cpu(),
                    name=["equilibrium", "longitudinal", "transversal"][k],
                    showlegend=i == 0, marker=dict(color=cmap[k])),
                row=i+1, col=1
            )
    fn = path_out.joinpath(name).with_suffix(".html").as_posix()
    fig.write_html(fn)


if __name__ == '__main__':
    sim_emc()
