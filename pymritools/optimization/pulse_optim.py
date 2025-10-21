import logging
import json

import torch
import numpy as np
import pathlib as plib
import tqdm

import plotly.graph_objects as go
import plotly.colors as plc
import plotly.subplots as psub
from matplotlib.pyplot import colorbar

from pymritools.config.rf import RFPulse
from pymritools.seqprog.core.events import RF
from pymritools.simulation.emc.sequence.pulse import PulseSimulationSettings, Pulse
from pymritools.config import setup_program_logging, setup_parser
from pymritools.simulation.emc.core import functions

from tests.utils import get_test_result_output_dir, ResultMode

logger = logging.getLogger(__name__)


def optimise_excitation_pulse(settings: PulseSimulationSettings):
    path = plib.Path(get_test_result_output_dir("pulse_optim_sinc_exc", mode=ResultMode.OPTIMIZATION)).absolute()
    settings.out_path = path.as_posix()
    settings.visualize = False
    settings.pulse_name = "excitation"

    # we set up the pulse simulation object
    pulse_sim = Pulse(settings=settings)

    slice_thickness = 0.0007        # [m]
    target_mag_xy = torch.zeros((pulse_sim.data.sample_axis.shape[0]), device=pulse_sim.device)
    target_ph_xy = torch.zeros((pulse_sim.data.sample_axis.shape[0]), device=pulse_sim.device)
    target_mag_xy[pulse_sim.data.sample_axis.abs() < slice_thickness / 2] = 1
    target_mag_z = pulse_sim.data.sample.clone().to(torch.float32)
    target_mag_z[pulse_sim.data.sample_axis.abs() < slice_thickness / 2] = 0

    # additionally introduce some target weighting towards the center
    target_m_weighting = torch.exp(-pulse_sim.data.sample_axis**2 / 0.005**2)
    # we want to emphasise rippling reduction
    target_m_weighting *= 2 / target_m_weighting.max()
    target_m_weighting[target_mag_xy > 0.5] = 1

    # assume we have a constant given gradient function
    # grad_init = torch.full_like(pulse_sim.grad_pulse.data_grad, -30.0) + torch.randn_like(pulse_sim.grad_pulse.data_grad) * 0.1
    grad_init = pulse_sim.grad_pulse.data_grad.clone()
    pulse_mask = torch.abs(pulse_sim.grad_pulse.data_pulse_x + 1j * pulse_sim.grad_pulse.data_pulse_y).squeeze() > 1e-12
    # grad = grad_init.clone().requires_grad_(True)
    # we need to adjust the simulation itself to optimise
    pulse_x_init = pulse_sim.grad_pulse.data_pulse_x.clone()
    # pulse_x_init = pulse_sim.grad_pulse.data_pulse_x.squeeze()[pulse_mask].clone()
    # pulse_y_init = pulse_sim.grad_pulse.data_pulse_y.clone()
    pulse_y_init = pulse_sim.grad_pulse.data_pulse_y.squeeze()[pulse_mask].clone()

    # pulse_x = pulse_x_init.clone().requires_grad_(True)
    pulse_y = pulse_y_init.clone().requires_grad_(True)

    # set some optimisation parameters
    max_num_iter = 50
    # lr = torch.full((max_num_iter,), 2e-8)
    lr = torch.linspace(2e-9, 8e-10, max_num_iter)

    losses = []
    losses_profile = []
    losses_sar = []

    pulse_iter =[]
    grad_iter = []
    mag_iter = []
    bar = tqdm.trange(max_num_iter)

    for idx in bar:
        # we set up the pulse simulation object
        pulse_sim = Pulse(settings=settings)
        # pulse
        # px = torch.zeros_like(pulse_sim.grad_pulse.data_pulse_x)
        # px[0, pulse_mask] = pulse_x
        py = torch.zeros_like(pulse_sim.grad_pulse.data_pulse_y)
        py[0, pulse_mask] = pulse_y
        data = functions.propagate_gradient_pulse_relax(
            pulse_x=pulse_x_init, pulse_y=py,
            grad=pulse_sim.grad_pulse.data_grad, sim_data=pulse_sim.data,
            dt_s=pulse_sim.grad_pulse.dt_sampling_steps_us * 1e-6
        )
        mag_exc = torch.squeeze(data.magnetization_propagation)[:, :3]
        m_cplx = mag_exc[:, 0] + 1j * mag_exc[:, 1]
        m_xy = torch.abs(m_cplx)
        p_xy = torch.angle(m_cplx)
        m_z = mag_exc[:, 2]
        mag_iter.append([m_xy, p_xy, m_z])

        # compute losses
        loss_profile = torch.nn.MSELoss()(m_xy * target_m_weighting, target_mag_xy) + torch.nn.MSELoss()(m_z, target_mag_z)
        losses_profile.append(loss_profile.item())

        loss_sar = torch.sum(torch.abs(pulse_y**2)) * 1e7
        losses_sar.append(loss_sar.item())

        loss = loss_profile + 0.67 * loss_sar
        losses.append(loss.item())

        bar.set_description(f"Loss: {loss.item():.3f}, SAR: {loss_sar.item():.3f}, Profile: {loss_profile.item():.3f}")
        loss.backward()
        with torch.no_grad():
            # pulse_x -= lr[idx] * pulse_x.grad
            pulse_y -= lr[idx] * pulse_y.grad
            # grad -= lr[idx] * grad.grad

            pulse_iter.append([pulse_x_init.clone().detach(), py.clone().detach()])
            # grad_iter.append(grad)

        # pulse_x.grad.zero_()
        pulse_y.grad.zero_()
        # grad.grad.zero_()

    plot_pulse_iter(
        pulse_iter=pulse_iter, path=path,
        p_init=[pulse_sim.grad_pulse.data_pulse_x, pulse_sim.grad_pulse.data_pulse_y, pulse_sim.grad_pulse.data_grad]
    )
    plot_losses(losses_profile=losses_profile, losses_sar=losses_sar, losses=losses, path=path)
    plot_mag_iter(
        mag_iter=mag_iter, path=path, sample_axis=pulse_sim.data.sample_axis,
        target_mag=[target_mag_xy, target_ph_xy, target_mag_z]
    )

    rf_optim = RFPulse(
        name="excitation",
        duration_in_us=2000,
        time_bandwidth=2.5,
        num_samples=pulse_y_init.shape[0],
        signal=pulse_y.detach().cpu().numpy()
    )
    save(
        rf_optim=rf_optim, losses=losses, losses_profile=losses_profile, losses_sar=losses_sar,
         mag_iter=mag_iter, pulse_iter=pulse_iter, path=path
    )


def optimise_refocusing_pulse(settings: PulseSimulationSettings):
    path = plib.Path(get_test_result_output_dir("pulse_optim_sinc_ref", mode=ResultMode.OPTIMIZATION)).absolute()
    settings.out_path = path.as_posix()
    settings.visualize = False
    settings.pulse_name = "refocusing"

    # we set up the pulse simulation object
    pulse_sim = Pulse(settings=settings)

    slice_thickness = 0.0007        # [m]

    target_mag_xy = torch.zeros((pulse_sim.data.sample_axis.shape[0]), device=pulse_sim.device)
    target_ph_xy = torch.zeros((pulse_sim.data.sample_axis.shape[0]), device=pulse_sim.device)
    target_mag_xy[pulse_sim.data.sample_axis.abs() < slice_thickness / 2] = 1
    target_mag_z = pulse_sim.data.sample.clone().to(torch.float32)
    target_mag_z[pulse_sim.data.sample_axis.abs() < slice_thickness / 2] = 0

    # additionally introduce some target weighting towards the center
    target_m_weighting = torch.exp(-pulse_sim.data.sample_axis**2 / 0.005**2)
    # we want to emphasise rippling reduction
    target_m_weighting *= 1.3 / target_m_weighting.max()
    target_m_weighting[target_mag_xy > 0.5] = 1

    # assume we have a constant given gradient function
    # grad_init = torch.full_like(pulse_sim.grad_pulse.data_grad, -30.0) + torch.randn_like(pulse_sim.grad_pulse.data_grad) * 0.1
    grad_init = pulse_sim.grad_pulse_ref.data_grad.clone()
    pulse_mask = torch.abs(pulse_sim.grad_pulse_ref.data_pulse_x + 1j * pulse_sim.grad_pulse_ref.data_pulse_y).squeeze() > 1e-12
    # grad = grad_init.clone().requires_grad_(True)
    # we need to adjust the simulation itself to optimise
    # pulse_x_init = pulse_sim.grad_pulse_ref.data_pulse_x.clone()
    pulse_x_init = pulse_sim.grad_pulse_ref.data_pulse_x.squeeze()[pulse_mask].clone()
    pulse_y_init = pulse_sim.grad_pulse_ref.data_pulse_y.clone()
    # pulse_y_init = pulse_sim.grad_pulse_ref.data_pulse_y.squeeze()[pulse_mask].clone()

    pulse_x = pulse_x_init.clone().requires_grad_(True)
    # pulse_y = pulse_y_init.clone().requires_grad_(True)

    # set some optimisation parameters
    max_num_iter = 50
    # lr = torch.full((max_num_iter,), 2e-8)
    lr = torch.linspace(2e-9, 8e-10, max_num_iter)

    losses = []
    losses_profile = []
    losses_sar = []

    pulse_iter =[]
    grad_iter = []
    mag_iter = []
    bar = tqdm.trange(max_num_iter)

    for idx in bar:
        # we set up the pulse simulation object
        pulse_sim = Pulse(settings=settings)
        # propagate excitation pulse
        # pulse
        data = functions.propagate_gradient_pulse_relax(
            pulse_x=pulse_sim.grad_pulse.data_pulse_x, pulse_y=pulse_sim.grad_pulse.data_pulse_y,
            grad=pulse_sim.grad_pulse.data_grad, sim_data=pulse_sim.data,
            dt_s=pulse_sim.grad_pulse.dt_sampling_steps_us * 1e-6
        )
        # delay
        relax_matrix = functions.matrix_propagation_relaxation_multidim(
            dt_s=0.00182, sim_data=pulse_sim.data
        )
        data = functions.propagate_matrix_mag_vector(relax_matrix, sim_data=data)

        # refocusing pulse
        px = torch.zeros_like(pulse_sim.grad_pulse_ref.data_pulse_x)
        px[0, pulse_mask] = pulse_x
        # py = torch.zeros_like(pulse_sim.grad_pulse_ref.data_pulse_y)
        # py[0, pulse_mask] = pulse_y
        data = functions.propagate_gradient_pulse_relax(
            pulse_x=px, pulse_y=pulse_y_init,
            grad=pulse_sim.grad_pulse_ref.data_grad, sim_data=data,
            dt_s=pulse_sim.grad_pulse_ref.dt_sampling_steps_us * 1e-6
        )

        # delay
        data = functions.propagate_matrix_mag_vector(relax_matrix, sim_data=data)
        mag_refocus = torch.squeeze(data.magnetization_propagation)[:, :3]
        m_cplx = mag_refocus[:, 0] + 1j * mag_refocus[:, 1]
        m_xy = torch.abs(m_cplx)
        p_xy = torch.angle(m_cplx)
        m_z = mag_refocus[:, 2]
        mag_iter.append([m_xy, p_xy, m_z])

        # compute losses
        loss_profile = torch.nn.MSELoss()(m_xy * target_m_weighting, target_mag_xy) + torch.nn.MSELoss()(m_z, target_mag_z)
        losses_profile.append(loss_profile.item())

        loss_sar = (torch.sum(torch.abs(pulse_x**2))) * 1e7
        losses_sar.append(loss_sar.item())

        loss = loss_profile + 0.15 * loss_sar
        losses.append(loss.item())

        bar.set_description(f"Loss: {loss.item():.3f}, SAR: {loss_sar.item():.3f}, Profile: {loss_profile.item():.3f}")
        loss.backward()
        with torch.no_grad():
            pulse_x -= lr[idx] * pulse_x.grad
            # pulse_y -= lr[idx] * pulse_y.grad
            # grad -= lr[idx] * grad.grad

            pulse_iter.append([px.clone().detach(), pulse_y_init.clone().detach()])
            # grad_iter.append(grad)

        pulse_x.grad.zero_()
        # pulse_y.grad.zero_()
        # grad.grad.zero_()

    plot_pulse_iter(
        pulse_iter=pulse_iter, path=path,
        p_init=[
            pulse_sim.grad_pulse_ref.data_pulse_x, pulse_sim.grad_pulse_ref.data_pulse_y,
            pulse_sim.grad_pulse_ref.data_grad
        ]
    )
    plot_losses(losses_profile=losses_profile, losses_sar=losses_sar, losses=losses, path=path)
    plot_mag_iter(
        mag_iter=mag_iter, path=path, sample_axis=pulse_sim.data.sample_axis,
        target_mag=[target_mag_xy, target_ph_xy, target_mag_z]
    )

    rf_optim = RFPulse(
        name="refocusing",
        duration_in_us=2500,
        time_bandwidth=2.5,
        num_samples=pulse_x_init.shape[0],
        signal=pulse_x.detach().cpu().numpy()
    )

    save(
        rf_optim=rf_optim, losses=losses, losses_profile=losses_profile, losses_sar=losses_sar,
         mag_iter=mag_iter, pulse_iter=pulse_iter, path=path
    )


def save(rf_optim, losses, losses_profile, losses_sar, mag_iter, pulse_iter, path):
    fn = path.joinpath("optim_pulse").with_suffix(".json")
    logger.info(f"Write File: {fn}")
    rf_optim.save_json(path=fn, indent=2)

    results = {
        "losses": losses,
        "losses_profile": losses_profile,
        "losses_sar": losses_sar
    }
    fn = path.joinpath("optimisation_losses").with_suffix(".json")
    logger.info(f"Write File: {fn}")
    with open(fn.as_posix(), mode="w") as f:
        json.dump(results, f, indent=2)

    fn = path.joinpath("optimisation_mag_iter").with_suffix(".pt")
    logger.info(f"Write File: {fn}")

    mag_iter = torch.stack([torch.stack(m) for m in mag_iter])
    torch.save(mag_iter, fn)

    fn = path.joinpath("optimisation_pulse_iter").with_suffix(".pt")
    logger.info(f"Write File: {fn}")
    pulse_iter = torch.stack([torch.stack(p) for p in pulse_iter])
    torch.save(pulse_iter, fn)


def plot_pulse_iter(pulse_iter: list, path: plib.Path, p_init=(None, None, None)):
    fig = psub.make_subplots(
        rows=3, cols=1,
        row_titles=["Pulse x", "Pulse y", "Grad z"],
    )
    cmap = plc.sample_colorscale("Inferno", len(pulse_iter), 0.1, 0.9)
    # for i in range(2):
    #     # plot weighting once
    #     fig.add_trace(
    #         go.Scatter(
    #             y=apo.cpu(),
    #             name=f"Target weighting",
    #             line=dict(color="violet", width=0),
    #             fill="tozeroy",
    #             opacity=0.3,
    #             mode="lines"
    #         ),
    #         row=1 + i, col=1
    #     )
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

    # put SLR init on top
    for j, pp in enumerate(p_init):
        pp = pp.squeeze()
        fig.add_trace(
            go.Scatter(
                y=pp.detach().cpu(),
                mode="lines",
                line=dict(color="teal"),
                showlegend=False,
                opacity=0.7
            ),
            row=j+1, col=1
        )
    fn = path.joinpath("pulse").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)


def plot_losses(losses_profile, losses_sar, losses, path):
    fig = psub.make_subplots(
        rows=3, cols=1,
        row_titles=["Profile", "SAR", "Total"],
        shared_xaxes=True, vertical_spacing=0.01
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

def plot_mag_iter(mag_iter: list, path: plib.Path, sample_axis: torch.Tensor, target_mag=(None, None, None)):
    # plot for reference
    cmap = plc.sample_colorscale("deep_r", len(mag_iter), 0.1, 0.9)
    fig = psub.make_subplots(
        rows=3, cols=1,
        row_titles=["mag xy", "ph xy", "z"],
        shared_xaxes=True,
        vertical_spacing=0.01
    )
    for i, m in enumerate(mag_iter):
        for j, mm in enumerate(m):
            # if i == 0:
                # plot weighting once
                # fig.add_trace(
                #     go.Scatter(
                #         x=pulse_sim.data.sample_axis.cpu(), y=target_m_weighting[j].cpu(),
                #         name=f"Target weighting",
                #         line=dict(color="violet", width=0),
                #         fill="tozeroy",
                #         opacity=0.3,
                #         mode="lines"
                #     ),
                #     row=1+j, col=1
                # )
            fig.add_trace(
                go.Scatter(
                    x=sample_axis.cpu(), y=mm.detach().cpu(),
                    name=[f"Mag. xy", "Phase xy", "Mag z"][j],
                    line=dict(color=cmap[i]),
                    mode="lines",
                    opacity = torch.linspace(0.2, 0.5, len(mag_iter)).tolist()[i],
                    showlegend=False
                ),
                row=1+j, col=1
            )
    for i, t in enumerate([mag_iter[0], target_mag]):
        targ = "Target " if i > 0 else "Init "
        for j, mm in enumerate(t):
            fig.add_trace(
                go.Scatter(
                    x=sample_axis.cpu(), y=mm.detach().cpu(),
                    name=f"{targ} {[f'Mag. xy', 'Phase xy', 'Mag z'][j]}",
                    line=dict(color=["teal", "salmon"][i]),
                    mode="lines"
                ),
                row=1+j, col=1
            )
    fn = path.joinpath("mag_optim").with_suffix(".html")
    logger.info(f"Write file: {fn}")
    fig.write_html(fn)


def plot_results(settings: PulseSimulationSettings):
    path_ref = plib.Path(get_test_result_output_dir("pulse_optim_slr_ref", mode=ResultMode.OPTIMIZATION)).absolute()
    path_exc = plib.Path(get_test_result_output_dir("pulse_optim_slr_exc", mode=ResultMode.OPTIMIZATION)).absolute()

    settings.out_path = path_ref.as_posix()
    settings.visualize = False
    settings.pulse_name = "excitation"

    # we set up the pulse simulation object
    pulse_sim = Pulse(settings=settings)

    pulse_iter_ref = torch.load(path_ref.joinpath("optimisation_pulse_iter").with_suffix(".pt"))
    pulse_iter_exc = torch.load(path_exc.joinpath("optimisation_pulse_iter").with_suffix(".pt"))

    # Create figure with secondary y-axis
    fig = psub.make_subplots(
        rows=2, cols=1,
        specs=[
            [{"secondary_y":True}],
            [{"secondary_y":True}]
        ]
    )
    for i, gp in enumerate([pulse_sim.grad_pulse, pulse_sim.grad_pulse_ref]):
        g = gp.data_grad
        fig.add_trace(
            go.Scatter(
                x=np.arange(g.shape[0])*5e-3, y=g.cpu(),
                showlegend=False,
                fill="tozeroy",
                mode="lines",
                line=dict(color="teal")
            ),
            row=1+i, col=1,
            secondary_y=False
        )
    cmap = plc.sample_colorscale("Inferno", pulse_iter_exc.shape[0], 0.1, 0.9)
    for i, pulse_iter in enumerate([pulse_iter_exc, pulse_iter_ref]):
        pulse_iter.squeeze_()
        for j, pp in enumerate(pulse_iter):
            ppp = pp[1-i%2]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(ppp.shape[0])*5e-3, y=ppp.cpu(),
                    showlegend=False,
                    mode="lines",
                    line=dict(color=cmap[j])
                ),
                row=1+i, col=1,
                secondary_y=True
            )
    # adjust midlines, RF
    fig.update_yaxes(
        row=1, col=1, secondary_y=True, range=(-6e-6, 6e-6),
        tickmode="array", tickvals=np.linspace(-5e-6, 5e-6, 5),
        title="RF amplitude [a.u.]"
    )
    fig.update_yaxes(
        row=2, col=1, secondary_y=True, range=(-7.5e-6, 7.5e-6),
        tickmode="array", tickvals=np.linspace(-6e-6, 6e-6, 7),
        title="RF amplitude [a.u.]"
    )
    # Grad
    fig.update_yaxes(
        row=1, col=1, secondary_y=False, range=(-120, 120),
        tickmode="array", tickvals=np.linspace(-100, 100, 5).astype(int),
        color="teal", title="g<sub>slice</sub> [mT/m]"
    )
    fig.update_yaxes(
        row=2, col=1, secondary_y=False, range=(-75, 75),
        tickmode="array", tickvals=np.linspace(-60, 60, 7).astype(int),
        color="teal", title="g<sub>slice</sub> [mT/m]"
    )

    fn = path_ref.joinpath("pulse_optim_combined").with_suffix(".html")
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
    # to not throw an error we set the output to the file, later we use the experiment file logic
    args.settings.out_path = plib.Path(__file__).absolute().as_posix()

    settings = PulseSimulationSettings.from_cli(args=args.settings)
    # set some additionals
    settings.visualize = False
    settings.kernel_file = plib.Path("/data/pt_np-jschmidt/code/PyMRItools/optimization/mese_slr_bpr5m_kernels.pkl").as_posix()
    settings.display()

    try:
        plot_results(settings=settings)
        # optimise_excitation_pulse(settings)
        # optimise_refocusing_pulse(settings)
    except Exception as e:
        parser.print_usage()
        logger.exception(e)
        exit(-1)


