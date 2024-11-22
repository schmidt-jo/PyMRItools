"""
This script is intended to optimize the pulse using some of the EMC bloch equation simulation
 functionality to calculate a slice profile and optimize this for a simulatneous multislice acquisition.
"""
import logging
import pathlib as plib

import numpy as np
import polars as pl
import torch
import tqdm
from scipy.constants import physical_constants
import plotly.graph_objects as go
import plotly.subplots as psub

from pymritools.config.emc import EmcSimSettings, EmcParameters, SimulationData
from pymritools.simulation.emc.core.blocks import GradPulse
from pymritools.simulation.emc.core.functions import (
    matrix_propagation_relaxation_multidim, propagate_gradient_pulse_relax,
    propagate_matrix_mag_vector, sum_sample_acquisition
)


def main():
    # setup logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
        datefmt='%I:%M:%S', level=logging.INFO
    )

    logging.info("Setup")
    rjust = 50
    # setup device
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info(f"device: {device}".rjust(rjust))

    # hardcode some of the path and parameters
    path = plib.Path("./optimization/sms/").absolute()
    gamma_pi = physical_constants["proton gyromag. ratio in MHz/T"][0] * 1e6 * 2 * torch.pi

    # take some stuff straight out the emc package
    # setup sample axis and sampling resolution
    # we want to aim for 0.7 mm slice thickness for 2 slices taken 5 cm apart, take 10 cm sample axis
    num_samples_across_slice = 4000
    extend_across_slice = 0.08  # m
    logging.info(f"spatial resolution: {extend_across_slice / num_samples_across_slice} m/sample".rjust(rjust))

    logging.info("Set simulation parameters")
    # params simulation
    T1 = 1.5    # s
    T2 = 35     # ms
    B1 = 1.0
    params = EmcParameters(etl=1, esp=1, sample_number=num_samples_across_slice, length_z=extend_across_slice/2)
    settings = EmcSimSettings(t1_list=[T1], t2_list=[T2], b1_list=[B1], resample_pulse_to_dt_us=1)
    sim_data = SimulationData(params=params, settings=settings, device=device)

    # build target profile
    target_slice_thickness = 0.0007     # m
    target_slice_shift = 0.04           # shift between slices
    target_shape_mag = torch.zeros_like(sim_data.sample_axis)
    target_shape_phase = torch.zeros_like(sim_data.sample_axis)
    for idx in range(2):
        mask = torch.logical_and(
            sim_data.sample_axis > pow(-1, idx) * target_slice_shift / 2 - target_slice_thickness / 2,
            sim_data.sample_axis < pow(-1, idx) * target_slice_shift / 2 + target_slice_thickness / 2
        )
        target_shape_mag[mask] = 1
        logging.info(f"number of samples per slice {idx+1}: {torch.count_nonzero(mask)}".rjust(rjust))

    # set pulse properties
    duration_pulse_us = 4000        # us
    duration_grad_z_us = 6000       # us
    dt_us = 2
    fa_deg = 90
    # °
    # use 1 us samplings for now, initialize pulse
    grad_z = torch.zeros(int(duration_grad_z_us / dt_us), dtype=torch.float32, device=device)
    rf_pulse = torch.zeros(int(duration_grad_z_us / dt_us), dtype=torch.complex64, device=device)

    # give it some initialized gaussian shape having slice selective gz and spoiling
    sig = duration_pulse_us/4
    rf_pulse[:int(duration_pulse_us/dt_us)] = 1 / np.sqrt(2 * np.pi * sig**2) * torch.exp(
        -(torch.arange(0, duration_pulse_us, dt_us) - duration_pulse_us/2)**2 / (2 * sig**2) + 1j * np.pi / 2
    )
    # normalize it to cause 90° excitation
    normalized_shape = rf_pulse / torch.linalg.norm(rf_pulse)
    # calculate flip angle
    flip_angle_normalized_shape = torch.sum(torch.abs(normalized_shape * gamma_pi)) * dt_us * 1e-6
    # scale to wanted fa
    rf_pulse = fa_deg / 180 * torch.pi * normalized_shape / flip_angle_normalized_shape
    # give some value to gz
    grad_z[:int(duration_pulse_us/dt_us)] = -12.0      # mT/m
    grad_z[int(duration_pulse_us/dt_us):] = -50.0        # mT/m

    # calculate the excitation already, is unchanged for the sim
    sim_data_exci = propagate_gradient_pulse_relax(
        pulse_x=rf_pulse[None].real, pulse_y=rf_pulse[None].imag, grad=grad_z,
        sim_data=sim_data, dt_s=dt_us*1e-6
    )

    fig = psub.make_subplots(
        rows=2, cols=3, shared_yaxes=True,
        column_titles=["Sample Z Magnetization", "Target XY Magnetization Profile", "Excitation XY Magnetization Profile"],
        specs=[
            [{}, {}, {}],
            [{"colspan": 3}, None, None]
        ]
    )
    fig.add_trace(
        go.Scatter(y=sim_data.sample_axis.cpu().numpy(), x=sim_data.sample.cpu().numpy(), fill="tozerox"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=sim_data.sample_axis.cpu().numpy(), x=target_shape_mag.cpu().numpy(), fill="tozerox", name="target magnetization"),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=sim_data.sample_axis.cpu().numpy(), x=target_shape_phase.cpu().numpy(), fill="tozerox", name="target phase"),
        row=1, col=2
    )
    excitation_profile = torch.squeeze(sim_data_exci.magnetization_propagation)
    excitation_profile_mag = torch.linalg.norm(excitation_profile[:, :2], dim=1)
    excitation_profile_phase = torch.arctan(excitation_profile[:, 1] / excitation_profile[:, 0]) / torch.pi
    fig.add_trace(
        go.Scatter(y=sim_data.sample_axis.cpu().numpy(), x=excitation_profile_mag.cpu().numpy(), fill="tozerox", name="magnetization"),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(y=sim_data.sample_axis.cpu().numpy(), x=excitation_profile_phase.cpu().numpy(), fill="tozerox", name="phase"),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(y=sim_data.sample_axis.cpu().numpy(), x=excitation_profile[:, 2].cpu().numpy(), fill="tozerox", name="z"),
        row=1, col=3
    )
    pulse_shape_mag = torch.abs(rf_pulse) / torch.max(torch.abs(rf_pulse))
    fig.add_trace(
        go.Scatter(x=torch.arange(duration_grad_z_us).cpu().numpy(), y=pulse_shape_mag.cpu().numpy(), name="norm. init pulse mag"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=torch.arange(duration_grad_z_us).cpu().numpy(), y=torch.angle(rf_pulse).cpu().numpy()/np.pi, name="norm. init pulse phase"),
        row=2, col=1
    )
    grad_shape = grad_z / torch.max(torch.abs(grad_z))
    fig.add_trace(
        go.Scatter(x=torch.arange(duration_grad_z_us).cpu().numpy(), y=grad_shape.cpu().numpy(), name="init grad_shape", fill="tozeroy"),
        row=2, col=1
    )

    fig.show()

    logging.info(f"Start optimization")
    # here we plug the shapes
    optim_pulse = rf_pulse.clone().requires_grad_(True)
    optim_grad_z = grad_z.clone().requires_grad_(True)

    # iterate
    losses = []
    max_num_iter = 50
    conv_count = 0
    bar = tqdm.trange(max_num_iter)
    for idx in bar:
        l_rate = 0.05 if idx < 10 else 0.01

        sim_data = SimulationData(
            params=params, settings=settings, device=device
        )
        # calculate the excitation already, is unchanged for the sim
        sim_data_exci = propagate_gradient_pulse_relax(
            pulse_x=optim_pulse[None].real, pulse_y=optim_pulse[None].imag, grad=optim_grad_z,
            sim_data=sim_data, dt_s=dt_us * 1e-6
        )

        # compute losses
        excitation_profile = torch.squeeze(sim_data_exci.magnetization_propagation)
        mag_profile = torch.linalg.norm(excitation_profile[:, :2], dim=1)
        phase_profile = torch.arctan(excitation_profile[:, 1] / excitation_profile[:, 0]) / torch.pi

        loss_target_mag = torch.linalg.norm(mag_profile - target_shape_mag)
        loss_target_phase = torch.linalg.norm(phase_profile - target_shape_phase)

        # minimize both distances, slight priority to magnetiozation
        loss = 10 * loss_target_mag + loss_target_phase
        loss.backward()
        with torch.no_grad():
            optim_pulse.sub_(l_rate*optim_pulse.grad)
            optim_grad_z.sub_(l_rate*optim_grad_z.grad)

        optim_pulse.grad.zero_()
        optim_grad_z.grad.zero_()


        pr = {
            "loss": f"{loss.item():.4f}",
            "mag": f"{loss_target_mag.item():.4f}", "phase": f"{loss_target_phase.item():.4f}",

        }
        bar.set_postfix(pr)
        losses.append(pr)

    losses = pl.DataFrame(losses)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            y=losses["snr"], name="snr"
        )
    )
    fig.add_trace(
        go.Scattergl(
            y=losses["sar"], name="sar"
        )
    )
    fig.add_trace(
        go.Scattergl(
            y=losses["loss"], name="loss"
        )
    )
    # path_fig = plib.Path("/data/pt_np-jschmidt/data/03_sequence_dev/mese_pulse_train_optimization/optimization/")
    # fig.write_html(path_fig.joinpath("optimization_losses").with_suffix(".html").as_posix())
    # losses.write_json(path_fig.joinpath("losses_df").with_suffix(".json").as_posix())
    fig.show()


if __name__ == '__main__':
    main()
