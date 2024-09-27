from pymritools.simulation.emc.core import functions, GradPulse
from .base import Simulation
from pymritools.config.emc import EmcParameters, EmcSimSettings, SimulationData
import torch
import tqdm
import logging

log_module = logging.getLogger(__name__)


class MEGESSE(Simulation):
    def __init__(self, params: EmcParameters, settings: EmcSimSettings):
        super().__init__(params=params, settings=settings)

    def _prep(self):
        """ want to set up gradients and pulses like in the megesse jstmc protocol
        For this we need all parts that are distinct and then set them up to push the calculation through
        """
        log_module.info("\t - MEGESSE sequence")
        log_module.info('\t - pulse gradient preparation')
        # excitation pulse
        self.gp_excitation = GradPulse.prep_grad_pulse_excitation(params=self.params, settings=self.settings)
        # its followed by the partial fourier readout GRE, if we dont sample the read and just use summing
        # we can assume the partial fourier is dealt with upon reconstruction.
        # hence we use just the acquisition with appropriate timing, and ignore read directions for now

        # built list of grad_pulse events, acquisition and timing
        self.gps_refocus = []
        for r_idx in torch.arange(self.params.etl):
            gp_refocus = GradPulse.prep_grad_pulse_refocus(
                params=self.params, refocus_pulse_number=r_idx, force_sym_spoil=True
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
        self.params.etl = 3 * self.rf_etl + 1
        # we have partial fourier in first readout
        self.partial_fourier_gre1: float = 3 / 4

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

        # after excitation
        # first readout echo time
        t_gre1 = (
                self.params.duration_excitation / 2 + self.params.duration_excitation_verse_lobes +
                self.params.duration_excitation_rephase +
                (self.partial_fourier_gre1 - 0.5) * self.params.duration_acquisition
        )
        # first set echo time
        t_gre1_set = self.params.tes[0] * 1e6
        # set var
        self.sequence_timings.register_timing(name="exc_gre1", value_us=t_gre1_set - t_gre1)

        # echo to refocusing
        t_gre1_ref1 = (
                self.params.duration_acquisition * 0.5 + self.params.duration_crush +
                self.params.duration_refocus_verse_lobes + self.params.duration_refocus / 2
        )
        # set midpoint gre1 til ref1 is half of se (tes[2]) time minus first echo
        t_gre1_ref1_set = 1e6 * (self.params.tes[2] / 2 - self.params.tes[0])
        # set var
        self.sequence_timings.register_timing(name="gre1_ref1", value_us=t_gre1_ref1_set - t_gre1_ref1)

        # refocusing to gradient readout
        t_ref1_gre2 = (
                self.params.duration_refocus / 2 + self.params.duration_refocus_verse_lobes +
                self.params.duration_crush + self.params.duration_acquisition / 2
        )
        # set midpoint second gradient echo is tes[1] minus refocusing time (half of spin echo time tes[2])
        t_ref_gre_set = 1e6 * (self.params.tes[1] - self.params.tes[2] / 2)
        self.sequence_timings.register_timing(name="ref_gre", value_us=t_ref_gre_set - t_ref1_gre2)

        # gradient echo 2 to spin echo (two halves of the acquisition duration
        t_gre2_se = self.params.duration_acquisition
        # set midpoints of echoes subtracted
        t_gre2_se_set = 1e6 * (self.params.tes[2] - self.params.tes[1])
        # set var
        self.sequence_timings.register_timing(name="gre_se", value_us=t_gre2_se_set - t_gre2_se)

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

        # sample partial fourier gradient echo readout
        # delay
        delay_relax = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("exc_gre1"), sim_data=self.data
        )
        self.data = functions.propagate_matrix_mag_vector(delay_relax, sim_data=self.data)

        # acquisition
        # take the sum of the contributions of the individual spins at pf readout echo time
        self.data = functions.sum_sample_acquisition(
            etl_idx=0, params=self.params, sim_data=self.data,
            acquisition_duration_s=self.params.duration_acquisition * 1e-6,
            partial_fourier=self.partial_fourier_gre1
        )
        if self.settings.visualize:
            # save excitation profile snapshot
            self.set_magnetization_profile_snap("gre1_post_acquisition")
        # delay
        delay_relax = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("gre1_ref1"), sim_data=self.data
        )
        self.data = functions.propagate_matrix_mag_vector(delay_relax, sim_data=self.data)

        # have only two timings left repeatedly, hence we can calculate the matrices already
        mat_prop_ref_gre_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("ref_gre"), sim_data=self.data
        )
        mat_prop_gre_se_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("gre_se"), sim_data=self.data
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
            self.data = functions.propagate_matrix_mag_vector(mat_prop_ref_gre_time, sim_data=self.data)
            # acquisition gre
            # take the sum of the contributions of the individual spins at central readout echo time
            self.data = functions.sum_sample_acquisition(
                etl_idx=1 + rf_idx * 3, params=self.params, sim_data=self.data,
                acquisition_duration_s=self.params.duration_acquisition * 1e-6
            )
            # delay between readouts
            self.data = functions.propagate_matrix_mag_vector(mat_prop_gre_se_time, sim_data=self.data)
            # acquisition se
            # take the sum of the contributions of the individual spins at central readout echo time
            self.data = functions.sum_sample_acquisition(
                etl_idx=2 + rf_idx * 3, params=self.params, sim_data=self.data,
                acquisition_duration_s=self.params.duration_acquisition * 1e-6
            )
            # delay between readouts
            self.data = functions.propagate_matrix_mag_vector(mat_prop_gre_se_time, sim_data=self.data)
            # acquisition gre
            # take the sum of the contributions of the individual spins at central readout echo time
            self.data = functions.sum_sample_acquisition(
                etl_idx=3 + rf_idx * 3, params=self.params, sim_data=self.data,
                acquisition_duration_s=self.params.duration_acquisition * 1e-6
            )
            # delay to pulse
            self.data = functions.propagate_matrix_mag_vector(mat_prop_ref_gre_time, sim_data=self.data)

            if self.settings.visualize:
                # save excitation profile snapshot
                self.set_magnetization_profile_snap(snap_name=f"refocus_{rf_idx + 1}_post_acquisition")

