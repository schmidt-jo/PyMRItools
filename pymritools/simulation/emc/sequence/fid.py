from .base import Simulation
from .. import blocks, functions
from pymritools.config.emc import EmcSettings, EmcParameters
import torch
import logging

log_module = logging.getLogger(__name__)


class FID(Simulation):
    def __init__(self, params: EmcParameters, settings: EmcSettings,
                 device: torch.device = torch.device("cpu"), num_mag_evol_plot: int = 3):
        super().__init__(params=params, settings=settings)

    def _prep(self):
        # setup single values only
        self.settings.t2_list = [50]
        self.settings.t1_list = [1.5]
        self.settings.b1_list = [0.8]
        self.params.etl = 1
        # no crushing - rephasing to calculate
        area_slice_select = self.params.gradient_excitation * self.params.duration_excitation
        area_rephase = - 0.5 * area_slice_select
        self.params.gradient_excitation_rephase = (area_rephase / self.params.duration_excitation_rephase)

        self.setup_simulation_data()

    def simulate(self):
        log_module.info("Simulating FID sequence")
        # get pulse
        gp_excitation = blocks.GradPulse.prep_single_grad_pulse(
            params=self.params, excitation_flag=True, grad_rephase_factor=1.05
        )
        gp_excitation.plot(sim_data=self.data)

        # get acquisition
        acquisition = blocks.GradPulse.prep_acquisition(params=self.params)
        # propagate pulse
        # pulse
        self.data = functions.propagate_gradient_pulse_relax(
            pulse_x=gp_excitation.data_pulse_x, pulse_y=gp_excitation.data_pulse_y,
            grad=gp_excitation.data_grad, sim_data=self.data,
            dt_s=gp_excitation.dt_sampling_steps_us * 1e-6
        )
        # # plot excitation profile
        # fig = plotting.plot_running_mag(fig, self.data, id=plot_idx)
        # plot_idx += 1
        # # sample acquisition
        # if fft:
        #     # prephase acquisition in case of fft
        #     pre_acq = blocks.GradPulse.prep_acquisition(params=self.params)
        #     pre_acq.data_grad = - pre_acq.data_grad
        #     pre_acq.duration = pre_acq.duration / 2
        #     self.data = functions.propagate_gradient_pulse_relax(
        #         pulse_x=pre_acq.data_pulse_x, pulse_y=pre_acq.data_pulse_y, grad=pre_acq.data_grad,
        #         sim_data=self.data, dt_s=pre_acq.duration * 1e-6
        #     )
        #     # plot dephased
        #     fig = plotting.plot_running_mag(fig, self.data, id=plot_idx)
        #     plot_idx += 1
        #     # acquisition
        #     self.data = functions.sample_acquisition(
        #         etl_idx=0, sim_params=self.params, sim_data=self.data,
        #         acquisition_grad=acquisition.data_grad, dt_s=acquisition.dt_sampling_steps * 1e-6
        #     )
        #     # plot after acquisition
        #     fig = plotting.plot_running_mag(fig, self.data, id=plot_idx)
        #     plot_idx += 1
        #     log_module.debug('Signal array processing fourier')
        #     image_tensor = torch.fft.ifftshift(
        #         torch.fft.ifft(self.data.signal_tensor, dim=-1)
        #     )
        #     self.data.emc_signal_mag = 2 * torch.sum(torch.abs(image_tensor),
        #                                             dim=-1) / self.params.settings.acquisition_number
        #     self.data.emc_signal_phase = 2 * torch.sum(torch.angle(image_tensor),
        #                                               dim=-1) / self.params.settings.acquisition_number
        #     plotting.plot_signal_traces(self.data)
        #     # rephase acquisition
        #     self.data = functions.propagate_gradient_pulse_relax(
        #         pulse_x=pre_acq.data_pulse_x, pulse_y=pre_acq.data_pulse_y, grad=pre_acq.data_grad,
        #         sim_data=self.data, dt_s=pre_acq.duration * 1e-6
        #     )
        # else:
        #     # propagate half of acquisition time - to equalize with above we need half the time prephaser
        #     # and half the time for read to arrive at the echo
        #     relax_matrix = functions.matrix_propagation_relaxation_multidim(
        #         dt_s=acquisition.duration * 1e-6, sim_data=self.data
        #     )
        #     self.data = functions.propagate_matrix_mag_vector(relax_matrix, self.data)
        #     # sum
        #     mag = torch.sum(self.data.magnetization_propagation[0, 0, 0], dim=0)
        #     self.data.emc_signal_mag[0, 0, 0] = torch.abs(mag[0] + 1j * mag[1])
        #     self.data.emc_signal_phase[0, 0, 0, 0] = torch.angle(mag[0] + 1j * mag[1])
        #     # again propagate relaxation to arrive at same state (second half of read and rephasing)
        #     self.data = functions.propagate_matrix_mag_vector(relax_matrix, self.data)
        # fig = plotting.plot_running_mag(fig, self.data, id=plot_idx)
        # plot_idx += 1
        # plotting.display_running_plot(fig, f"fid_simulation_fft-{fft}")
