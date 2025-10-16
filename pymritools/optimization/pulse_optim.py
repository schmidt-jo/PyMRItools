import logging
import torch
import pathlib as plib
import tqdm

import plotly.graph_objects as go

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

    # set some optimisation parameters
    max_num_iter = 10
    lr = torch.full((max_num_iter,), 0.001)

    losses = []
    losses_profile = []
    losses_sar = []

    pulse_iter =[]
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

        # compute losses
        loss_profile = torch.linalg.norm(m_xy - target_m_xy) + torch.linalg.norm(m_z - target_m_z)
        losses_profile.append(loss_profile.item())

        loss_sar = torch.linalg.norm(pulse_x) + torch.linalg.norm(pulse_y)
        losses_sar.append(loss_sar.item())

        loss = loss_profile + 0.001 * loss_sar
        losses.append(loss.item())

        bar.set_description(f"Loss: {loss.item():.3f}, SAR: {loss_sar.item():.3f}, Profile: {loss_profile.item():.3f}")
        loss.backward()
        with torch.no_grad():
            pulse_x -= lr[idx] * pulse_x.grad
            pulse_y -= lr[idx] * pulse_y.grad

            pulse_iter.append([pulse_x, pulse_y])

        pulse_x.grad.zero_()
        pulse_y.grad.zero_()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=losses,
        )
    )
    fig.write_html(path.joinpath("losses").with_suffix(".html"))

    # plot for reference
    fig = go.Figure()
    for i, t in enumerate([target_m_xy, target_m_z, m_xy, m_z]):
        targ = "Target " if i < 2 else ""
        fig.add_trace(
            go.Scatter(
                x=pulse_sim.data.sample_axis.cpu(), y=t.detach().cpu(),
                name=f"{targ}Mag. {['xy', 'z'][i % 2]}"
            )
        )
    fig.write_html(path.joinpath("mag_optim").with_suffix(".html"))

if __name__ == '__main__':
    # setup logging
    setup_program_logging(name="EMC simulation - Pulse", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="EMC simulation - Pulse",
        dict_config_dataclasses={"settings": PulseSimulationSettings}
    )

    settings = PulseSimulationSettings.from_cli(args=args.settings)
    settings.display()

    try:
        optimise_excitation_pulse(settings)
    except Exception as e:
        parser.print_usage()
        logger.exception(e)
        exit(-1)
