import logging
import torch
import pathlib as plib
import tqdm

import plotly.graph_objects as go
import plotly.colors as plc
import plotly.subplots as psub

from pymritools.simulation.emc.sequence.pulse import PulseSimulationSettings, Pulse
from pymritools.config import setup_program_logging, setup_parser
from pymritools.simulation.emc.core import functions

from tests.utils import get_test_result_output_dir, ResultMode

logger = logging.getLogger(__name__)


def optimise_excitation_pulse(settings: PulseSimulationSettings):
    path = plib.Path(get_test_result_output_dir("pulse_optim_exc", mode=ResultMode.OPTIMIZATION)).absolute()
    settings.out_path = path.as_posix()
    settings.visualize = False

    # we set up the pulse simulation object
    pulse_sim = Pulse(settings=settings)

    # we set a magnetisation target shape
    slice_thickness = 0.0007        # [m]
    target_m_xy = torch.zeros_like(pulse_sim.data.sample_axis)
    target_m_xy[pulse_sim.data.sample_axis.abs() < slice_thickness / 2] = 1
    target_m_z = pulse_sim.data.sample.clone()
    target_m_z[pulse_sim.data.sample_axis.abs() < slice_thickness / 2] = 0

    # assume we have a constant given gradient function
    # we need to adjust the simulation itself to optimise
    pulse_x = pulse_sim.grad_pulse.data_pulse_x.clone().requires_grad_(True)
    pulse_y = pulse_sim.grad_pulse.data_pulse_y.clone().requires_grad_(True)
    pulse_iter =[[pulse_x, pulse_y]]

    loss_sar = torch.sum(torch.abs(pulse_x ** 2)) + torch.sum(torch.abs(pulse_y ** 2))
    losses_sar = [loss_sar.item()]

    # set some optimisation parameters
    max_num_iter = 10
    # lr = torch.full((max_num_iter,), 1e-11)
    lr = torch.linspace(1e-10, 1e-12, max_num_iter)

    losses = []
    losses_profile = []

    mag_iter = []
    bar = tqdm.trange(max_num_iter)

    for idx in bar:
        # we set up the pulse simulation object
        pulse_sim = Pulse(settings=settings)
        # pulse
        data = functions.propagate_gradient_pulse_relax(
            pulse_x=pulse_x, pulse_y=pulse_y,
            grad=pulse_sim.grad_pulse.data_grad, sim_data=pulse_sim.data,
            dt_s=pulse_sim.grad_pulse.dt_sampling_steps_us * 1e-6
        )
        mag_exc = torch.squeeze(data.magnetization_propagation)
        m_xy = torch.linalg.norm(mag_exc[:, :2], dim=1)
        m_z = mag_exc[:, 2]
        mag_iter.append([m_xy, m_z])

        # compute losses
        loss_profile = torch.linalg.norm(m_xy - target_m_xy) + torch.linalg.norm(m_z - target_m_z)
        losses_profile.append(loss_profile.item())

        loss_sar = torch.sum(torch.abs(pulse_x**2)) + torch.sum(torch.abs(pulse_y**2)) * 1e9
        losses_sar.append(loss_sar.item())

        loss = loss_profile + loss_sar
        losses.append(loss.item())

        bar.set_description(f"Loss: {loss.item():.3f}, SAR: {loss_sar.item():.3f}, Profile: {loss_profile.item():.3f}")
        loss.backward()
        with torch.no_grad():
            pulse_x -= lr[idx] * pulse_x.grad
            pulse_y -= lr[idx] * pulse_y.grad

            pulse_iter.append([pulse_x.clone().detach(), pulse_y.clone().detach()])

        pulse_x.grad.zero_()
        pulse_y.grad.zero_()

    fig = psub.make_subplots(
        rows=2, cols=1,
        row_titles=["Pulse x", "Pulse y"],
    )
    cmap = plc.sample_colorscale("Inferno", len(pulse_iter), 0.1, 0.9)
    for i, p in enumerate(pulse_iter):
        for j, pp in enumerate(p):
            pp = pp.squeeze()
            fig.add_trace(
                go.Scatter(
                    y=pp.detach().cpu(),
                    mode="lines",
                    line=dict(color=cmap[i]),
                    showlegend=False
                ),
                row=j+1, col=1
            )

    fn = path.joinpath("pulse").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    fig = psub.make_subplots(
        rows=3, cols=1,
        row_titles=["Profile", "SAR", "Total"],
        shared_xaxes=True
    )
    for i, l in enumerate([losses_profile, losses_sar, losses]):
        fig.add_trace(
            go.Scatter(
                y=l,
            ),
            row=i+1, col=1
        )
    fn = path.joinpath("losses").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)

    # plot for reference
    cmap = [
        plc.sample_colorscale("deep_r", len(mag_iter), 0.1, 0.9),
        plc.sample_colorscale("matter_r", len(mag_iter), 0.1, 0.9)
        ]
    fig = go.Figure()
    for i, m in enumerate(mag_iter):
        for j, mm in enumerate(m):
            fig.add_trace(
                go.Scatter(
                    x=pulse_sim.data.sample_axis.cpu(), y=mm.detach().cpu(),
                    name=f"Mag. {['xy', 'z'][j % 2]}",
                    line=dict(color=cmap[j][i]),
                    mode="lines",
                    opacity = 0.3,
                    showlegend=False
                )
            )
    for i, t in enumerate([target_m_xy, target_m_z]):
        targ = "Target " if i < 2 else ""
        fig.add_trace(
            go.Scatter(
                x=pulse_sim.data.sample_axis.cpu(), y=t.detach().cpu(),
                name=f"{targ}Mag. {['xy', 'z'][i % 2]}",
                line=dict(color=["teal", "salmon"][i]),
                mode="lines"
            )
        )
    fn = path.joinpath("mag_optim").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)


if __name__ == '__main__':
    # setup logging
    setup_program_logging(name="EMC simulation - Pulse", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="EMC simulation - Pulse",
        dict_config_dataclasses={"settings": PulseSimulationSettings}
    )

    settings = PulseSimulationSettings.from_cli(args=args.settings)
    settings.visualize = False
    settings.display()

    try:
        optimise_excitation_pulse(settings)
    except Exception as e:
        parser.print_usage()
        logger.exception(e)
        exit(-1)
