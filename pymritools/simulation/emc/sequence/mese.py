import time
import logging
import pathlib as plib

import pickle
import json
import tqdm
import torch

from pymritools.config import setup_program_logging, setup_parser
from pymritools.simulation.emc.core import functions, GradPulse
from pymritools.config.emc import EmcSimSettings, EmcParameters
from pymritools.simulation.emc.sequence.base_sequence import Simulation

log_module = logging.getLogger(__name__)


class MESE(Simulation):
    def __init__(self, params: EmcParameters, settings: EmcSimSettings):
        super().__init__(params=params, settings=settings)

    def _prep_from_kernels(self, kernel_file: plib.Path):
        with open(kernel_file, "rb") as f:
            kernels = pickle.load(f)
        # load tes
        te_file = plib.Path(self.settings.te_file).absolute()
        if not te_file.is_file():
            err = f"TE file {te_file} does not exist"
            log_module.error(err)
            raise FileNotFoundError(err)
        with open(te_file, "r") as f:
            self.tes = json.load(f)

        if self.settings.visualize:
            kf = self.fig_path.joinpath("kernels")
            kf.mkdir(exist_ok=True)
            for k, v in kernels.items():
                v.plot(path=kf, name=k, file_suffix="png")

        # excitation
        k = kernels["excitation"]
        # build grad pulse
        self.gp_excitation = GradPulse.prep_from_pulseq_kernel(
            kernel=k, name="excitation", device=self.device, b1s=self.data.b1_vals, flip_angle_rad=torch.pi/2,
        )
        # fill params info
        self.params.duration_excitation = k.rf.t_duration_s * 1e6
        self.params.duration_excitation_rephase = (k.grad_slice.get_duration() - k.rf.t_delay_s - k.rf.t_duration_s) * 1e6

        # refocus
        k = kernels["refocus_1"]
        self.gps_refocus = [
            GradPulse.prep_from_pulseq_kernel(
                kernel=k, name="refocus", pulse_number=0, device=self.device, b1s=self.data.b1_vals,
                flip_angle_rad=self.params.refocus_angle[0] / 180 * torch.pi
            )
        ]
        k = kernels["refocus"]
        self.gps_refocus.extend([
            GradPulse.prep_from_pulseq_kernel(
                kernel=k, name="refocus", pulse_number=rfi+1, device=self.device, b1s=self.data.b1_vals,
                flip_angle_rad=self.params.refocus_angle[rfi+1] / 180 * torch.pi
            ) for rfi in range(self.params.etl-1)
        ])
        # fill params info
        self.params.duration_refocus = k.rf.t_duration_s * 1e6
        self.params.duration_crush = (k.grad_slice.get_duration() - k.rf.t_duration_s) * 1e6 / 2

        # extract params from acquisition kernel
        self.params.acquisition_number = 1
        self.params.bw = 1 / kernels["acq"].adc.get_duration()
        self.gp_acquisition = GradPulse.prep_acquisition(params=self.params)
        # prep sim data due to etl change - we spawn on cpu and use the GPU memory in the batching
        # self.data = SimulationData(params=self.params, settings=self.settings, device=torch.device("cpu"))

    def _prep(self):
        log_module.info("\t - MESE sequence")
        # prep pulse grad data - this holds the pulse data and timings
        log_module.info('\t - pulse gradient preparation')
        # load kernels
        kernel_file = plib.Path(self.settings.kernel_file).absolute()
        if kernel_file.is_file():
            msg = f"\t - Loading GradPulse data from Kernel file {kernel_file}"
            log_module.info(msg)
            self._prep_from_kernels(kernel_file=kernel_file)
        else:
            msg = f"\t -Loading GradPulse data from parameter arguments. (no kernel file provided)"
            log_module.info(msg)
            self._prep_from_params()

        if self.settings.visualize:
            log_module.info("\t - plot grad pulse data")
            self.gp_excitation.plot(b1_vals=self.data.b1_vals, fig_path=self.fig_path)
            self.gps_refocus[0].plot(b1_vals=self.data.b1_vals, fig_path=self.fig_path)
            self.gps_refocus[1].plot(b1_vals=self.data.b1_vals, fig_path=self.fig_path)
            self.gp_acquisition.plot(b1_vals=self.data.b1_vals, fig_path=self.fig_path)

    def _prep_from_params(self):
        self.gp_excitation = GradPulse.prep_grad_pulse_excitation(
            pulse=self.pulse, params=self.params, settings=self.settings,
            b1_vals=self.data.b1_vals,
        )

        gp_refocus_1 = GradPulse.prep_grad_pulse_refocus(
            pulse=self.pulse, params=self.params, settings=self.settings, refocus_pulse_number=0,
            b1_vals=self.data.b1_vals,
        )

        # built list of grad_pulse events, acquisition and timing
        self.gps_refocus = [gp_refocus_1]
        for r_idx in torch.arange(1, self.params.etl):
            gp_refocus = GradPulse.prep_grad_pulse_refocus(
                pulse=self.pulse, params=self.params, settings=self.settings, refocus_pulse_number=r_idx,
                b1_vals=self.data.b1_vals,
            )
            self.gps_refocus.append(gp_refocus)


    def _set_device(self):
        # set devices
        self.gp_excitation.set_device(self.device)
        self.sequence_timings.set_device(self.device)
        for gp in self.gps_refocus:
            gp.set_device(self.device)
        self.gp_acquisition.set_device(self.device)

    def _register_sequence_timings(self):
        log_module.info(f"\t - calculate sequence timing")
        # all in [us]
        # before first refocusing
        time_pre_pulse = 1000 * self.params.esp / 2 - (
                self.params.duration_excitation / 2 + self.params.duration_excitation_verse_lobes +
                self.params.duration_excitation_rephase +
                self.params.duration_refocus / 2 + self.params.duration_refocus_verse_lobes
        )
        self.sequence_timings.register_timing(name="pre_refocus_1", value_us=time_pre_pulse)

        # after first refocusing
        time_post_pulse = 1000 * self.params.esp / 2 - (
                self.params.duration_refocus / 2 + self.params.duration_refocus_verse_lobes +
                self.params.duration_crush + self.params.duration_acquisition / 2
        )
        self.sequence_timings.register_timing(name="post_refocus_1", value_us=time_post_pulse)

        self.sequence_timings.register_timing(name="pre_post_refocus", value_us=time_post_pulse)

    def _simulate(self):
        """ want to set up gradients and pulses like in the mese standard protocol
        For this we need all parts that are distinct and then set them up to pulse the calculation through
        """
        log_module.info("__ Simulating MESE Sequence __")
        t_start = time.time()
        # --- starting sim matrix propagation --- #
        log_module.info("calculate matrix propagators 1/4")
        # excitation
        self.data = functions.propagate_gradient_pulse_relax(
            pulse_x=self.gp_excitation.data_pulse_x,
            pulse_y=self.gp_excitation.data_pulse_y,
            grad=self.gp_excitation.data_grad,
            sim_data=self.data, dt_s=self.gp_excitation.dt_sampling_steps_us * 1e-6
        )
        if self.settings.visualize:
            # save excitation profile snapshot
            self.set_magnetization_profile_snap(snap_name="excitation")

        # calculate timing matrices (there are only 3)
        log_module.info("calculate matrix propagators 2/4")
        # first refocus
        mat_prop_ref1_pre_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("pre_refocus_1"), sim_data=self.data
        )
        log_module.info("calculate matrix propagators 3/4")
        mat_prop_ref1_post_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s(1), sim_data=self.data
        )
        # other refocus
        log_module.info("calculate matrix propagators 4/4")
        mat_prop_ref_pre_post_time = functions.matrix_propagation_relaxation_multidim(
            dt_s=self.sequence_timings.get_timing_s("pre_post_refocus"), sim_data=self.data
        )
        log_module.debug("loop through refocusing")
        with tqdm.trange(self.params.etl) as t:
            t.set_description(f"processing sequence, refocusing pulse loop")
            for loop_idx in t:
                # timing
                if loop_idx == 0:
                    m_p_pre = mat_prop_ref1_pre_time
                else:
                    m_p_pre = mat_prop_ref_pre_post_time
                self.data = functions.propagate_matrix_mag_vector(m_p_pre, sim_data=self.data)
                # pulse
                self.data = functions.propagate_gradient_pulse_relax(
                    pulse_x=self.gps_refocus[loop_idx].data_pulse_x,
                    pulse_y=self.gps_refocus[loop_idx].data_pulse_y,
                    grad=self.gps_refocus[loop_idx].data_grad,
                    sim_data=self.data,
                    dt_s=self.gps_refocus[loop_idx].dt_sampling_steps_us * 1e-6
                )
                # timing
                if loop_idx == 0:
                    m_p_post = mat_prop_ref1_post_time
                else:
                    m_p_post = mat_prop_ref_pre_post_time
                self.data = functions.propagate_matrix_mag_vector(m_p_post, sim_data=self.data)

                if self.settings.visualize:
                    # save profile snapshot after pulse
                    self.set_magnetization_profile_snap(snap_name=f"refocus_{loop_idx+1}_post_pulse")

                # acquisition
                if self.settings.signal_fourier_sampling:
                    # sample the acquisition with the readout gradient moved to into the slice direction
                    self.data = functions.sample_acquisition(
                        etl_idx=loop_idx, params=self.params, sim_data=self.data,
                        acquisition_grad=self.gp_acquisition.data_grad,
                        dt_s=self.gp_acquisition.dt_sampling_steps_us * 1e-6
                    )
                    # basically this gives us a signal evolution in k-space binned into spatial resolution bins
                else:
                    # take the sum of the contributions of the individual spins at central readout echo time
                    self.data = functions.sum_sample_acquisition(
                        etl_idx=loop_idx, params=self.params, sim_data=self.data,
                        acquisition_duration_s=self.params.duration_acquisition * 1e-6
                    )

                if self.settings.visualize:
                    # save excitation profile snapshot
                    self.set_magnetization_profile_snap(snap_name=f"refocus_{loop_idx + 1}_post_acquisition")

        if self.settings.signal_fourier_sampling:
            log_module.debug('Signal array processing fourier')
            image_tensor = torch.fft.fftshift(
                torch.fft.ifft(
                    torch.fft.ifftshift(self.data.signal_tensor, dim=-1),
                    dim=-1
                ),
                dim=-1
            )
            # if self.settings.visualize:
            #     # plot slice sampling image tensor (binned slice profile)
            #     plotting.plot_slice_img_tensor(slice_image_tensor=image_tensor, sim_data=self.data,
            #                                    out_path=self.fig_path)

            self.data.emc_signal_mag = 2 * torch.abs(torch.sum(image_tensor, dim=-1)) / self.params.acquisition_number
            self.data.emc_signal_phase = torch.angle(torch.sum(image_tensor, dim=-1))

        self.data.time = time.time() - t_start


def simulate(settings: EmcSimSettings, params: EmcParameters) -> None:
    sequence = MESE(params=params, settings=settings)
    sequence.simulate()
    sequence.save()


def main():
    # setup logging
    setup_program_logging(name="EMC simulation - MESE", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="EMC simulation - MESE",
        dict_config_dataclasses={"settings": EmcSimSettings, "params": EmcParameters}
    )

    settings = EmcSimSettings.from_cli(args=args.settings)
    settings.display()
    if settings.emc_params_file:
        params = EmcParameters.load(settings.emc_params_file)
    else:
        params = args.params

    try:
        simulate(settings=settings, params=params)
    except Exception as e:
        parser.print_usage()
        log_module.exception(e)
        exit(-1)


if __name__ == '__main__':
    main()

