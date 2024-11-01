import json
import logging
import pathlib as plib
import pickle

import tqdm
import torch

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.emc import EmcParameters, EmcSimSettings, SimulationData
from pymritools.config.rf import RFPulse
from pymritools.simulation.emc.sequence.base_sequence import Simulation
from pymritools.simulation.emc.core import functions, GradPulse

log_module = logging.getLogger(__name__)


class MEGESSE(Simulation):
    def __init__(self, params: EmcParameters, settings: EmcSimSettings):
        super().__init__(params=params, settings=settings)

    def _prep(self):
        # load kernels
        kernel_file = plib.Path(self.settings.kernel_file).absolute()
        if not kernel_file.is_file():
            err = f"Kernel file {kernel_file} does not exist"
            log_module.error(err)
            raise FileNotFoundError(err)
        with open(kernel_file, "rb") as f:
            kernels = pickle.load(f)
        # load tes
        te_file = plib.Path(self.settings.te_file).absolute()
        if not te_file.is_file():
            err = f"Kernel file {te_file} does not exist"
            log_module.error(err)
            raise FileNotFoundError(err)
        with open(te_file, "r") as f:
            te = json.load(f)
        self.params.tes = te

        if self.settings.visualize:
            for k, v in kernels.items():
                v.plot(path=self.fig_path, name=k, file_suffix="png")

        self.rf_etl = 4
        self.params.etl = 5 * self.rf_etl + 2

        # excitation
        k = kernels["excitation"]
        # build grad pulse
        self.gp_excitation = GradPulse.prep_from_pulseq_kernel(
            kernel=k, name="excitation", settings=self.settings, b1s=self.data.b1_vals, flip_angle_rad=torch.pi/2,
        )
        # fill params info
        self.params.duration_excitation = k.rf.t_duration_s * 1e6
        self.params.duration_excitation_rephase = (k.grad_slice.get_duration() - k.rf.t_delay_s - k.rf.t_duration_s) * 1e6

        # refocus
        k = kernels["refocus"]
        self.gps_refocus = [
            GradPulse.prep_from_pulseq_kernel(
                kernel=k, name="refocus", pulse_number=rfi, settings=self.settings, b1s=self.data.b1_vals,
                flip_angle_rad=140 / 180 * torch.pi
            ) for rfi in range(self.rf_etl)
        ]
        # fill params info
        self.params.duration_refocus = k.rf.t_duration_s * 1e6
        self.params.duration_crush = (k.grad_slice.get_duration() - k.rf.t_duration_s) * 1e6 / 2

        # extract params from acquisition kernel
        self.params.acquisition_number = 1
        self.params.bw = 1 / kernels["acq_bu"].adc.get_duration()
        self.gp_acquisition = GradPulse.prep_acquisition(params=self.params)
        # prep sim data due to etl change
        self.data = SimulationData(params=self.params, settings=self.settings, device=self.device)

        if self.settings.visualize:
            self.gp_excitation.plot(self.data.b1_vals, fig_path=self.fig_path)
            self.gps_refocus[0].plot(self.data.b1_vals, fig_path=self.fig_path)
            self.gps_refocus[1].plot(self.data.b1_vals, fig_path=self.fig_path)
            self.gp_acquisition.plot(self.data.b1_vals, fig_path=self.fig_path)


    def _prep_arxv(self):
        """ want to set up gradients and pulses like in the megesse protocol
        For this we need all parts that are distinct and then set them up to push the calculation through
        """
        log_module.info("\t - MEGESSE symmetrical sequence")
        log_module.info('\t - pulse gradient preparation')
        # excitation pulse
        self.gp_excitation = GradPulse.prep_grad_pulse_excitation(
            params=self.params, settings=self.settings, pulse=self.pulse
        )

        # built list of grad_pulse events, acquisition and timing
        self.gps_refocus = []
        for r_idx in torch.arange(self.params.etl):
            gp_refocus = GradPulse.prep_grad_pulse_refocus(
                pulse=self.pulse, params=self.params, settings=self.settings,
                refocus_pulse_number=r_idx, force_sym_spoil=True
            )
            self.gps_refocus.append(gp_refocus)

        if self.settings.visualize:
            self.gp_excitation.plot(self.data, fig_path=self.fig_path)
            self.gps_refocus[0].plot(self.data, fig_path=self.fig_path)
            self.gps_refocus[1].plot(self.data, fig_path=self.fig_path)
            self.gp_acquisition.plot(self.data, fig_path=self.fig_path)
        # for megesse the etl is number of refocussing pulses, not number of echoes,
        # we need to reset the simulation data with adopted etl
        # 1 gre, and then for every pulse in the etl there are 3 echoes -> etl = 3*etl + 1
        self.rf_etl = self.params.etl
        self.params.etl = 5 * self.rf_etl + 2

        # prep sim data due to etl change
        self.data = SimulationData(params=self.params, settings=self.settings, device=self.device)

    def _set_device(self):
        # set devices
        self.sequence_timings.set_device(self.device)
        self.gp_excitation.set_device(self.device)
        for gp in self.gps_refocus:
            gp.set_device(self.device)
        self.gp_acquisition.set_device(self.device)

    def _register_sequence_timings(self):
        log_module.info(f"\t - calculate sequence timing")
        # all in [us]
        # post excitation. We calculate the time the events take and compare with the set echo times.
        # this way we can fill any gaps with relaxation if needed
        # from excitation mid til first gre:
        t_gre1 = (
                self.params.duration_excitation / 2 + self.params.duration_excitation_verse_lobes +
                self.params.duration_excitation_rephase + self.params.duration_acquisition / 2
        )
        # first set echo time = time of first gre
        t_gre1_set = self.params.tes[0] * 1e6
        # set delay if needed: will happen after excitation pulse
        t_delay = t_gre1_set - t_gre1
        if t_delay < 0:
            err = f"timing between two grad pulses found to be negative to match prescribed echo times."
            log_module.error(err)
            raise ValueError(err)
        self.sequence_timings.register_timing(name="exc_e1", value_us=t_delay)

        # calculate time from acquisition mid to next acquisition of gre 2
        t_e2e = self.params.duration_acquisition
        # compare with difference of echo times
        t_gre2_set = (self.params.tes[1] - self.params.tes[0]) * 1e6
        t_delay = t_gre2_set - t_e2e
        if t_delay < 0:
            err = f"timing between two grad pulses found to be negative to match prescribed echo times."
            log_module.error(err)
            raise ValueError(err)
        self.sequence_timings.register_timing(name="e1_e2", value_us=t_delay)

        # calculate time from echo to refocus pulse (and since symmetric, the other way around)
        t_e2ref = (
            self.params.duration_acquisition * 0.5 + self.params.duration_crush +
            self.params.duration_refocus_verse_lobes + self.params.duration_refocus / 2
        )
        # compare with timing of the rf mid. we can calculate this from the echo times. it should be half way to se.
        # se is the 5th echo
        t_gre2_ref1_set = 1e6 * (self.params.tes[4] / 2 - self.params.tes[1])
        # set delay
        t_delay = t_gre2_ref1_set - t_e2ref
        if t_delay < 0:
            err = f"timing between two grad pulses found to be negative to match prescribed echo times."
            log_module.error(err)
            raise ValueError(err)
        self.sequence_timings.register_timing(name="e2_ref1", value_us=t_delay)

        # calculated time from ref to next echo, its the same as from echo to ref
        # compare with the time between ref and next echo set by te. remember ref mid is half of se time
        t_ref1_gre3_set = 1e6 * (self.params.tes[2] - self.params.tes[4] / 2)
        # set delay
        t_delay = t_ref1_gre3_set - t_e2ref
        if t_delay < 0:
            err = f"timing between two grad pulses found to be negative to match prescribed echo times."
            log_module.error(err)
            raise ValueError(err)
        self.sequence_timings.register_timing(name="ref_e3", value_us=t_delay)

        # already calculated time between echoes
        # compare with adjacent gre readouts
        t_e2e_set = 1e6 * (self.params.tes[3] - self.params.tes[2])
        # set delay
        t_delay = t_e2e_set - t_e2e
        if t_delay < 0:
            err = f"timing between two grad pulses found to be negative to match prescribed echo times."
            log_module.error(err)
            raise ValueError(err)
        self.sequence_timings.register_timing(name="e2e", value_us=t_delay)

        # already calculated echo to ref
        # compare with next ref (upon first ref the delay before the refocusing could be asymmetric).
        # next ref is at 1.5 * se time, take difference to previous echo
        t_e2ref_set = 1e6 * (1.5 * self.params.tes[4] - self.params.tes[6])
        t_delay = t_e2ref_set - t_e2ref
        if t_delay < 0:
            err = f"timing between two grad pulses found to be negative to match prescribed echo times."
            log_module.error(err)
            raise ValueError(err)
        self.sequence_timings.register_timing(name="e2ref", value_us=t_delay)

    def _simulate(self):
        if self.settings.signal_fourier_sampling:
            # not yet implemented, # ToDo
            err = "signal fourier sampling not yet implemented"
            log_module.error(err)
            raise AttributeError(err)

        log_module.info(f"Simulating MEGESSE sequence")
        # --- starting sim matrix propagation --- #
        log_module.info("calculate matrix propagation")
        # excitation
        self.data = functions.propagate_gradient_pulse_relax(
            pulse_x=self.gp_excitation.data_pulse_x, pulse_y=self.gp_excitation.data_pulse_y,
            grad=self.gp_excitation.data_grad,
            sim_data=self.data, dt_s=self.gp_excitation.dt_sampling_steps_us * 1e-6
        )
        if self.settings.visualize:
            # save excitation profile snapshot
            self.set_magnetization_profile_snap("excitation")

        # sample first gradient echo readouts
        # delay
        delay_relax = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("exc_e1"), sim_data=self.data
        )
        self.data = functions.propagate_matrix_mag_vector(delay_relax, sim_data=self.data)

        # acquisition
        # take the sum of the contributions of the individual spins at echo readout time
        self.data = functions.sum_sample_acquisition(
            etl_idx=0, params=self.params, sim_data=self.data,
            acquisition_duration_s=self.params.duration_acquisition * 1e-6,
        )
        if self.settings.visualize:
            # save excitation profile snapshot
            self.set_magnetization_profile_snap("gre1_post_acquisition")

        delay_relax = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("e1_e2"), sim_data=self.data
        )
        self.data = functions.propagate_matrix_mag_vector(delay_relax, sim_data=self.data)
        # take the sum of the contributions of the individual spins at echo readout time
        self.data = functions.sum_sample_acquisition(
            etl_idx=1, params=self.params, sim_data=self.data,
            acquisition_duration_s=self.params.duration_acquisition * 1e-6,
        )
        if self.settings.visualize:
            # save excitation profile snapshot
            self.set_magnetization_profile_snap("gre2_post_acquisition")

        # delay
        delay_relax = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("e2_ref1"), sim_data=self.data
        )
        self.data = functions.propagate_matrix_mag_vector(delay_relax, sim_data=self.data)

        # have only two timings left repeatedly, hence we can calculate the matrices already
        mat_prop_e2e_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("e2e"), sim_data=self.data
        )
        mat_prop_e2ref_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("e2ref"), sim_data=self.data
        )
        # rf loop
        for rf_idx in tqdm.trange(self.rf_etl, desc="processing sequence, refocusing pulse loop"):
            # propagate pulse
            self.data = functions.propagate_gradient_pulse_relax(
                pulse_x=self.gps_refocus[rf_idx].data_pulse_x,
                pulse_y=self.gps_refocus[rf_idx].data_pulse_y,
                grad=self.gps_refocus[rf_idx].data_grad,
                sim_data=self.data,
                dt_s=self.gps_refocus[rf_idx].dt_sampling_steps_us * 1e-6
            )
            if self.settings.visualize:
                # save profile snapshot after pulse
                self.set_magnetization_profile_snap(snap_name=f"refocus_{rf_idx + 1}_post_pulse")

            # timing from ref to gre
            self.data = functions.propagate_matrix_mag_vector(mat_prop_e2ref_time, sim_data=self.data)

            # acquisitions
            for idx_acq in range(5):
                # take the sum of the contributions of the individual spins at central readout echo time
                self.data = functions.sum_sample_acquisition(
                    etl_idx=2 + idx_acq + rf_idx * 5, params=self.params, sim_data=self.data,
                    acquisition_duration_s=self.params.duration_acquisition * 1e-6
                )
                # delay between readouts
                self.data = functions.propagate_matrix_mag_vector(mat_prop_e2e_time, sim_data=self.data)

            # delay to pulse
            self.data = functions.propagate_matrix_mag_vector(mat_prop_e2ref_time, sim_data=self.data)

            if self.settings.visualize:
                # save excitation profile snapshot
                self.set_magnetization_profile_snap(snap_name=f"refocus_{rf_idx + 1}_post_acquisitions")


def simulate(settings: EmcSimSettings, params: EmcParameters) -> None:
    sequence = MEGESSE(params=params, settings=settings)
    sequence.simulate()
    sequence.save()


def main():
    # setup logging
    setup_program_logging(name="EMC simulation - MEGESSE asymmetric echoes", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="EMC simulation - MEGESSE asymmetric echoes",
        dict_config_dataclasses={"settings": EmcSimSettings, "params": EmcParameters}
    )

    settings = EmcSimSettings.from_cli(args=args.settings)
    settings.display()

    params = args.params

    try:
        simulate(settings=settings, params=params)
    except Exception as e:
        parser.print_usage()
        log_module.exception(e)
        exit(-1)


if __name__ == '__main__':
    main()
