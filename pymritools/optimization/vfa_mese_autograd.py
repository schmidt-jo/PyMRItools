"""
This script is intended to optimize the mese (or megesse) pulse trains to produce dictionary curves with
minimal SAR while offering maximal SNR.
"""
import logging
import pathlib as plib
import pickle
import json

import polars as pl
import torch
import tqdm
from scipy.constants import physical_constants
import plotly.graph_objects as go

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

    logging.info("Setup parameters")
    # setup device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # hardcode some of the path and parameters
    path = plib.Path(
        "/data/pt_np-jschmidt/data/03_sequence_dev/build_sequences/2024-11-06_mese_megesse/mese_acc-4_min_spoil/"
    )
    gamma_pi = physical_constants["proton gyromag. ratio in MHz/T"][0] * 1e6 * 2 * torch.pi

    # only fill the needed ones
    params = EmcParameters(
        etl=7, esp=7.37, bw=350,
        sample_number=500, length_z=0.005,
        acquisition_number=1
    )
    settings = EmcSimSettings(
        t2_list=[[10, 100, 25], [100, 1000, 100]],
        b1_list=[[0.3, 1.7, 0.25]],
    )
    sim_data = SimulationData(
        params=params, settings=settings, device=device
    )

    # get blocks from kernel files
    with open(
            path.joinpath("mese_v1p0_acc-4p0_res-0p70-0p70-0p70_kernels").with_suffix(".pkl").as_posix(), "rb"
    ) as j_file:
        kernels = pickle.load(j_file)

    logging.info("Prepare GradPulse objects")
    # excitation
    gp_excitation = GradPulse.prep_from_pulseq_kernel(
        kernel=kernels["excitation"], name="excitation", b1s=sim_data.b1_vals, device=device,
        flip_angle_rad=torch.pi/2, dt_set_sampling_steps_us=10
    )
    # get the data
    pulse_excitation = gp_excitation.data_pulse_x + 1j * gp_excitation.data_pulse_y
    grad_excitation_z = gp_excitation.data_grad
    dt_excitation_us = gp_excitation.dt_sampling_steps_us
    t_excitation_us = torch.arange(grad_excitation_z.shape[0]) * dt_excitation_us

    # ref 1
    gp_refocus_1 = GradPulse.prep_from_pulseq_kernel(
        kernel=kernels["refocus_1"], name="refocus_1", b1s=sim_data.b1_vals, device=device,
        flip_angle_rad=torch.pi, dt_set_sampling_steps_us=10
    )
    # get the data
    pulse_ref1 = gp_refocus_1.data_pulse_x + 1j * gp_refocus_1.data_pulse_y
    grad_ref1_z = gp_refocus_1.data_grad
    dt_ref1_us = gp_refocus_1.dt_sampling_steps_us
    t_ref1_us = torch.arange(grad_ref1_z.shape[0]) * dt_ref1_us

    # refs
    gp_refocus = GradPulse.prep_from_pulseq_kernel(
        kernel=kernels["refocus"], name="refocus", b1s=sim_data.b1_vals, device=device,
        flip_angle_rad=torch.pi, dt_set_sampling_steps_us=10
    )
    # get the data
    pulse_ref = gp_refocus.data_pulse_x + 1j * gp_refocus.data_pulse_y
    grad_ref_z = gp_refocus.data_grad
    dt_ref_us = gp_refocus.dt_sampling_steps_us
    t_ref_us = torch.arange(grad_ref_z.shape[0]) * dt_ref_us

    # get tes
    with open(
        path.joinpath("mese_v1p0_acc-4p0_res-0p70-0p70-0p70_te").with_suffix(".json").as_posix(), "r"
    ) as j_file:
        tes_us = 1e3 * torch.tensor(json.load(j_file))

    logging.info("Calculate Timing")
    t_exci_mid_us = t_excitation_us[torch.max(torch.abs(pulse_excitation), dim=-1).indices[0]]
    t_ref1_mid_us = t_ref1_us[torch.max(torch.abs(pulse_ref1), dim=-1).indices[0]]
    t_ref_mid_us = t_ref_us[torch.max(torch.abs(pulse_ref), dim=-1).indices[0]]

    t_total_excitation_us = gp_excitation.duration_us
    t_total_refocus_1_us = gp_refocus_1.duration_us
    t_total_refocus_us = gp_refocus.duration_us

    # we need the refocusing 1 mid point to be half of the te, calculate how much time the grads already took
    # time from mid excitation to mid ref 1
    t_diff_exc_ref1_us = t_total_excitation_us - t_exci_mid_us + t_ref1_mid_us
    t_set_exc_ref1_us = tes_us[0] / 2
    t_delay_exc_ref1_s = (t_set_exc_ref1_us - t_diff_exc_ref1_us) * 1e-6
    if t_delay_exc_ref1_s < 0:
        err = f"found negative delay between excitation and refocus 1. check code."
        logging.error(err)
        raise ValueError(err)

    # time from mid ref1 to acquisition
    t_diff_ref1_acq_us = t_total_refocus_1_us - t_ref1_mid_us
    t_set_ref1_acq_us = tes_us[0]
    t_delay_ref1_acq_s = (t_set_ref1_acq_us - t_diff_ref1_acq_us) * 1e-6
    if t_delay_ref1_acq_s < 0:
        err = f"found negative delay between refocus 1 and acquisition. check code."
        logging.error(err)
        raise ValueError(err)

    # time from acquisition to ref, should from now on be symmetrical (also actually after ref1)
    t_diff_ref_acq_us = t_total_refocus_us - t_ref_mid_us
    t_set_ref_acq_us = (tes_us[1] - tes_us[0]) / 2
    t_delay_ref_acq_s = (t_set_ref_acq_us - t_diff_ref_acq_us) * 1e-6
    if t_delay_ref_acq_s < 0:
        err = f"found negative delay between refocus and acquisition. check code."
        logging.error(err)
        raise ValueError(err)

    logging.info("Calculate matrix propagators")
    relax_exc_ref1 = matrix_propagation_relaxation_multidim(
        dt_s=t_delay_exc_ref1_s, sim_data=sim_data
    )
    relax_ref1_acq = matrix_propagation_relaxation_multidim(
        dt_s=t_delay_ref1_acq_s, sim_data=sim_data
    )
    relax_ref_acq = matrix_propagation_relaxation_multidim(
        dt_s=t_delay_ref_acq_s, sim_data=sim_data
    )
    # prep excitation pulse as its not changing
    fa_excitation = torch.pi / 2
    p_e_normalized_shape = pulse_excitation / torch.linalg.norm(torch.abs(pulse_excitation), dim=-1, keepdim=True)
    p_e_fa_norm_shape = torch.sum(torch.abs(p_e_normalized_shape * gamma_pi), dim=-1) * dt_excitation_us * 1e-6
    pulse_excitation = fa_excitation / p_e_fa_norm_shape[:, None] * sim_data.b1_vals[:, None] * p_e_normalized_shape

    # calculate the excitation already, is unchanged for the sim
    sim_data_exci = propagate_gradient_pulse_relax(
        pulse_x=pulse_excitation.real, pulse_y=pulse_excitation.imag, grad=grad_excitation_z,
        sim_data=sim_data, dt_s=dt_excitation_us*1e-6
    )
    # propagate relaxation til ref 1
    sim_data_exci = propagate_matrix_mag_vector(
        relax_exc_ref1, sim_data=sim_data_exci
    )

    # and make them a factor - if we normalize the pulses to fa pi = 180 degrees, we can just multiply with fas
    # normalize the shapes s.th. we only need to multiply with fa
    fa_ref1 = torch.pi
    p_ref1_normalized_shape = pulse_ref1 / torch.linalg.norm(torch.abs(pulse_ref1), dim=-1, keepdim=True)
    p_ref1_fa_norm_shape = torch.sum(torch.abs(p_ref1_normalized_shape * gamma_pi), dim=-1) * dt_ref1_us * 1e-6
    pulse_ref1 = fa_ref1 / p_ref1_fa_norm_shape[:, None] * sim_data.b1_vals[:, None] * p_ref1_normalized_shape
    # normalize the shapes s.th. we only need to multiply with fa
    fa_ref = torch.pi
    p_ref_normalized_shape = pulse_ref / torch.linalg.norm(torch.abs(pulse_ref), dim=-1, keepdim=True)
    p_ref_fa_norm_shape = torch.sum(torch.abs(p_ref_normalized_shape * gamma_pi), dim=-1) * dt_ref_us * 1e-6
    pulse_ref = fa_ref / p_ref_fa_norm_shape[:, None] * sim_data.b1_vals[:, None] * p_ref_normalized_shape

    logging.info(f"Start optimization")
    # here we plug the actual flip angles
    fa_1 = torch.full((1,), 140.0/180.0, requires_grad=True, device=device)
    fa_2 = torch.full((1,), 140.0/180.0, requires_grad=True, device=device)
    fa_3 = torch.full((1,), 140.0/180.0, requires_grad=True, device=device)
    fa_4 = torch.full((1,), 140.0/180.0, requires_grad=True, device=device)
    fa_5 = torch.full((1,), 140.0/180.0, requires_grad=True, device=device)
    fa_6 = torch.full((1,), 140.0/180.0, requires_grad=True, device=device)
    fa_7 = torch.full((1,), 140.0/180.0, requires_grad=True, device=device)
    fas = torch.full((7,), 140.0/180.0)
    last_fas = fas.clone()
    # and make them a factor - if we normalize the pulses to fa pi = 180 degrees, we can just multiply with fas

    # iterate
    losses = []
    max_num_iter = 100
    conv_count = 0
    bar = tqdm.trange(max_num_iter)
    for idx in bar:
        l_rate = 0.05 if idx < 10 else 0.01

        sim_data = SimulationData(
            params=params, settings=settings, device=device
        )
        # calculate the excitation already, is unchanged for the sim
        sim_data_exci = propagate_gradient_pulse_relax(
            pulse_x=pulse_excitation.real, pulse_y=pulse_excitation.imag, grad=grad_excitation_z,
            sim_data=sim_data, dt_s=dt_excitation_us*1e-6
        )
        # propagate relaxation til ref 1
        sim_data_exci = propagate_matrix_mag_vector(
            relax_exc_ref1, sim_data=sim_data_exci
        )
        # do ref 1
        pulse_r1 = pulse_ref1.clone() * fa_1
        sim_data = propagate_gradient_pulse_relax(
            pulse_x=pulse_r1.real, pulse_y=pulse_r1.imag, grad=grad_ref1_z,
            sim_data=sim_data_exci, dt_s=dt_ref1_us*1e-6
        )
        # relax delay
        sim_data = propagate_matrix_mag_vector(
            propagation_matrix=relax_ref1_acq, sim_data=sim_data
        )
        # acquisition - has virtually no duration since we calculated the delay previously already
        sim_data = sum_sample_acquisition(
            etl_idx=0, params=params, sim_data=sim_data, acquisition_duration_s=1e-6
        )

        for idx_rf in torch.arange(1, params.etl):
            # delay
            sim_data = propagate_matrix_mag_vector(
                propagation_matrix=relax_ref_acq, sim_data=sim_data
            )
            if idx_rf == 1:
                fa = fa_2
            elif idx_rf == 2:
                fa = fa_3
            elif idx_rf == 3:
                fa = fa_4
            elif idx_rf == 4:
                fa = fa_5
            elif idx_rf == 5:
                fa = fa_6
            elif idx_rf == 6:
                fa = fa_7
            pulse = pulse_ref.clone() * fa
            # do rf
            sim_data = propagate_gradient_pulse_relax(
                pulse_x=pulse.real, pulse_y=pulse.imag, grad=grad_ref_z,
                sim_data=sim_data, dt_s=dt_ref_us * 1e-6
            )

            # delay
            sim_data = propagate_matrix_mag_vector(
                propagation_matrix=relax_ref_acq, sim_data=sim_data
            )

            # acquisition
            sim_data = sum_sample_acquisition(
                etl_idx=idx_rf, params=params, sim_data=sim_data, acquisition_duration_s=1e-6
            )

        # compute losses
        sar = torch.sqrt(
            (torch.pi * fa_1)**2 + (torch.pi * fa_2)**2 + (torch.pi * fa_3)**2 +
            (torch.pi * fa_4)**2 + (torch.pi * fa_5)**2 + (torch.pi * fa_6)**2 + (torch.pi * fa_7)**2
        )
        snr = torch.linalg.norm(sim_data.signal_mag, dim=-1).flatten().sum() / sim_data.total_num_sim
        # minimize sar, maximize snr, with a minimizing total loss
        loss = sar - 20 * snr
        loss.backward()
        with torch.no_grad():
            fa_1.data.sub_(l_rate*fa_1.grad.data)
            fa_2.data.sub_(l_rate*fa_2.grad.data)
            fa_3.data.sub_(l_rate*fa_3.grad.data)
            fa_4.data.sub_(l_rate*fa_4.grad.data)
            fa_5.data.sub_(l_rate*fa_5.grad.data)
            fa_6.data.sub_(l_rate*fa_6.grad.data)
            fa_7.data.sub_(l_rate*fa_7.grad.data)

        fa_1.grad.zero_()
        fa_2.grad.zero_()
        fa_3.grad.zero_()
        fa_4.grad.zero_()
        fa_5.grad.zero_()
        fa_6.grad.zero_()
        fa_7.grad.zero_()
        fas = torch.tensor([
            fa_1.detach().clone(), fa_2.detach().clone(), fa_3.detach().clone(), fa_4.detach().clone(),
            fa_5.detach().clone(), fa_6.detach().clone(), fa_7.detach().clone()
        ])
        convergence = torch.linalg.norm(fas - last_fas)
        last_fas = fas.detach().clone()

        pr = {
            "loss": f"{loss.item():.4f}", "sar": f"{sar.item():.4f}", "snr": f"{snr.item():.4f}",
            "convergence": f"{convergence:.4f}"
        }
        for i, f in enumerate(fas):
            pr.__setitem__(f"f_{i+1}", f"{f*180:.1f}")
        bar.postfix = pr
        losses.append(pr)
        if convergence < 1e-3:
            conv_count += 1
            if conv_count > 3:
                logging.info(f"reached convergence at iteration {idx}")
                break

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
    path_fig = plib.Path("/data/pt_np-jschmidt/data/03_sequence_dev/mese_pulse_train_optimization/optimization/")
    fig.write_html(path_fig.joinpath("optimization_losses").with_suffix(".html").as_posix())
    losses.write_json(path_fig.joinpath("losses_df").with_suffix(".json").as_posix())


if __name__ == '__main__':
    main()
