import logging
import pickle

import torch
import pathlib as plib

import plotly.graph_objects as go
import plotly.colors as plc
import plotly.subplots as psub

from pymritools.simulation.emc.sequence import MESE
from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.emc import EmcSimSettings, EmcParameters

from tests.utils import get_test_result_output_dir, ResultMode

logger = logging.getLogger(__name__)


def simulate_pulse_profiles(settings: EmcSimSettings, params: EmcParameters, force_sim: bool = False,
                            path: plib.Path = None):
    if path is None:
        path = plib.Path(
            get_test_result_output_dir(
                "simulate_pulse_profiles_slr_sp3300_msl_tb2p8", mode=ResultMode.EXPERIMENT
            )
        ).absolute()
    else:
        path = plib.Path(path)
    settings.out_path = path.as_posix()
    settings.visualize = True

    fn = path.joinpath("pulse_profiles").with_suffix(".pkl")
    if not fn.is_file() or force_sim:
        # reset simulation data to capture only one set of variables
        # here we fix some simple simulation parameters, see _prep
        # setup single values only
        settings.t2_list = [1000.0]
        settings.t1_list = [2.0]
        settings.b1_list = [1.0]

        # set up sequence simulation object
        mese = MESE(params=params, settings=settings)
        mese.simulate()

        # extract pulse profiles
        profiles = mese._fig_magnetization_profile_snaps
        axis = mese.data.sample_axis.cpu()
        profiles.append({"name": "sample_axis", "profile": axis})
        # save profiles
        logger.info(f"Save profiles to {fn.as_posix()}")
        with open(path.joinpath("pulse_profiles").with_suffix(".pkl"), "wb") as f:
            pickle.dump(profiles, f)
    else:
        logger.info(f"Load profiles from {fn.as_posix()}")
        with open(fn.as_posix(), "rb") as f:
            profiles = pickle.load(f)

    # build torch representation
    tprofs = []
    names = []
    for i, profile in enumerate(profiles):
        if profile["name"] in ["initial", "excitation"]:
            tprofs.append(profile["profile"])
            names.append(profile["name"][:4])
        elif "post_acquisition" in profile["name"]:
            n = profile["name"].split("post")[0][:-1]
            names.append(n[-1])
            tprofs.append(profile["profile"])
    ax = profiles[-1]["profile"]
    tprofs = torch.stack(tprofs)

    # plot post pulse profiles
    slice_thickness = 0.0007
    gap = 65
    fig = psub.make_subplots(
        rows=tprofs.shape[0], cols=1,
        row_titles=names,
        shared_xaxes=True
    )
    target = torch.zeros_like(tprofs[:2, :, 0])
    target[0, ax.abs() < slice_thickness / 2] = 1
    target[1, ax.abs() < (1 + gap/100) * slice_thickness / 2] = 1
    for i, profile in enumerate(tprofs):
        for j in range(2):
            fig.add_trace(
                go.Scatter(
                    x=ax * 1e3, y=target[j],
                    fill="tozeroy",
                    mode="lines",
                    showlegend=False,
                    opacity=0.8 if j == 0 else 0.1,
                    line=dict(color=["teal", "rgba(0, 102, 102, 0.05)"][j], width=1 if j == 0 else 0)
                ),
                row=i + 1, col=1
            )
        xy = torch.linalg.norm(profile[:, :2], dim=1)
        z = profile[:, 2] / profile[:, 2].abs().max()
        for j, m in enumerate([xy, z]):
            fig.add_trace(
                go.Scatter(
                    x=ax*1e3, y=m,
                    fill=None,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=["purple", "salmon"][j])
                ),
                row=i+1, col=1
            )
    fig.update_yaxes(range=(-1.2, 1.2))
    fign = path.joinpath("pulse_profiles").with_suffix(".html")
    logger.info(f"Write file: {fign}")
    fig.write_html(fign)


if __name__ == '__main__':
    # setup logging
    setup_program_logging(name="EMC simulation - MESE pulse profile", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="EMC simulation - MESE pulse profile",
        dict_config_dataclasses={"settings": EmcSimSettings, "params": EmcParameters}
    )

    settings = EmcSimSettings.from_cli(args=args.settings)
    settings.display()
    if settings.emc_params_file:
        params = EmcParameters.load(settings.emc_params_file)
    else:
        params = args.params

    try:
        simulate_pulse_profiles(settings=settings, params=params)
    except Exception as e:
        parser.print_usage()
        logger.exception(e)
        exit(-1)