import logging
import pathlib as plib

import tqdm
import torch
import pickle
import json
import numpy as np

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.emc import EmcParameters, EmcSimSettings, SimulationData
from pymritools.simulation.emc.sequence.base_sequence import Simulation
from pymritools.simulation.emc.core import functions, GradPulse

log_module = logging.getLogger(__name__)


class MEGESSE(Simulation):
    def __init__(self, params: EmcParameters, settings: EmcSimSettings):
        self.num_echoes_per_rf: int = 5
        super().__init__(params=params, settings=settings)

    def _prep_from_kernels(self, kernel_file: plib.Path):
        with open(kernel_file, "rb") as f:
            kernels = pickle.load(f)
        # load tes
        te_file = plib.Path(self.settings.te_file).absolute()
        if not te_file.is_file():
            err = f"Kernel file {te_file} does not exist"
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
            ) for rfi in range(self.rf_etl-1)
        ])
        # fill params info
        self.params.duration_refocus = k.rf.t_duration_s * 1e6
        self.params.duration_crush = (k.grad_slice.get_duration() - k.rf.t_duration_s) * 1e6 / 2

        # extract params from acquisition kernel
        self.params.acquisition_number = 1
        self.params.bw = 1 / kernels["acq_bu"].adc.get_duration()
        self.gp_acquisition = GradPulse.prep_acquisition(params=self.params)
        # prep sim data due to etl change - we spawn on cpu and use the GPU memory in the batching
        # self.data = SimulationData(params=self.params, settings=self.settings, device=torch.device("cpu"))

    def _prep_from_params(self):
        # excitation pulse
        self.gp_excitation = GradPulse.prep_grad_pulse_excitation(
            pulse=self.pulse, params=self.params, settings=self.settings,
            b1_vals=self.data.b1_vals,
        )
        # its followed by the partial fourier readout GRE, if we dont sample the read and just use summing
        # we can assume the partial fourier is dealt with upon reconstruction.
        # hence we use just the acquisition with appropriate timing, and ignore read directions for now

        # built list of grad_pulse events, acquisition and timing
        self.gps_refocus = []
        for r_idx in torch.arange(self.params.etl):
            gp_refocus = GradPulse.prep_grad_pulse_refocus(
                pulse=self.pulse, params=self.params, settings=self.settings,
                b1_vals=self.data.b1_vals, refocus_pulse_number=r_idx
            )
            self.gps_refocus.append(gp_refocus)

    def _prep(self):
        """ want to set up gradients and pulses like in the megesse jstmc protocol
        For this we need all parts that are distinct and then set them up to push the calculation through
        """
        log_module.info("\t - MEGESSE sequence")
        log_module.info('\t - pulse gradient preparation')
        self.rf_etl = self.params.etl
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
            self.gp_excitation.plot(b1_vals=self.data.b1_vals, fig_path=self.fig_path)
            self.gps_refocus[0].plot(b1_vals=self.data.b1_vals, fig_path=self.fig_path)
            self.gps_refocus[1].plot(b1_vals=self.data.b1_vals, fig_path=self.fig_path)
            self.gp_acquisition.plot(b1_vals=self.data.b1_vals, fig_path=self.fig_path)

        # for megesse the etl is number of refocussing pulses, not number of echoes,
        # we need to reset the simulation data with adopted etl
        # 1 gre, and then for every pulse in the etl there are 3 echoes -> etl = 3*etl + 1
        self.params.etl = self.num_echoes_per_rf * self.rf_etl

        # prep sim data due to etl change - we spawn on cpu and use the GPU memory in the batching
        self.data = SimulationData(params=self.params, settings=self.settings, device=torch.device("cpu"))

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

        # between excitation and first refocus
        t_exc_ref1 = 1e6 * self.params.tes[0] / 2 - (
            self.params.duration_excitation / 2 + self.params.duration_excitation_verse_lobes +
            self.params.duration_excitation_rephase +
            self.params.duration_refocus_verse_lobes + self.params.duration_refocus / 2
        )

        # between first refocus and echo = generally between echoes and refocusing
        t_ref_e = 1e6 * self.params.tes[0] / 2 - (
            self.params.duration_refocus_verse_lobes + self.params.duration_refocus / 2 +
            self.params.duration_crush + self.params.duration_acquisition * 0.5
        )

        # between readouts (should be constant)
        delta_tes = torch.diff(torch.tensor(self.params.tes)).unique()
        t_e2e = min(delta_tes) * 1e6

        # set var
        self.sequence_timings.register_timing(name="exc_ref1", value_us=t_exc_ref1)

        self.sequence_timings.register_timing(name="ref_e", value_us=t_ref_e)

        self.sequence_timings.register_timing(name="e_e", value_us=t_e2e)

    def _simulate(self):
        if self.settings.signal_fourier_sampling:
            # not yet implemented, # ToDo
            err = "signal fourier sampling not yet implemented"
            log_module.error(err)
            raise AttributeError(err)

        log_module.info(f"Simulating MEGESSE sequence")

        # need batch processing, take 10 t2 values at a time
        batch_size_t2 = 10
        num_batches = int(np.ceil(self.data.num_t2s / batch_size_t2))
        # expand allocated mag vector
        self.data.magnetization_propagation = self.data.magnetization_propagation.expand(
            (self.data.num_t1s, self.data.num_t2s, self.data.num_b1s, self.data.num_b0s, -1, -1)
        ).clone().contiguous()
        # ToDo: calculate GPU RAM dependencie and batch total number of values
        for nb in tqdm.trange(num_batches, desc="Batch Processing"):
            # --- starting sim matrix propagation --- #

            # spawn data with less t2 params -
            # spawn at cpu (we produce bigger tensors here and shrink them afterwards, no need to put all on GPU)
            data = SimulationData(params=self.params, settings=self.settings, device=torch.device("cpu"))

            start = nb * batch_size_t2
            end = min((nb + 1) * batch_size_t2, self.data.num_t2s)
            data.t2_vals = self.data.t2_vals[start:end]
            data.num_t2s = end - start
            # init tensors on device
            data.init_tensors(
                etl=self.params.etl,
                acquisition_number=self.params.acquisition_number,
                gamma_hz=self.params.gamma_hz
            )
            # additionally put tensors on device
            data.set_device(self.device)

            # calculate propagation matrices with respective t2 vals
            log_module.debug("calculate matrix propagation 1/2")
            # have only two timings left repeatedly, hence we can calculate the matrices already
            mat_prop_ref_e_time = functions.matrix_propagation_relaxation_multidim(
                dt_s=self.sequence_timings.get_timing_s("ref_e"), sim_data=data
            )
            log_module.debug("calculate matrix propagation 2/2")
            mat_prop_e_e_time = functions.matrix_propagation_relaxation_multidim(
                dt_s=self.sequence_timings.get_timing_s("e_e"), sim_data=data
            )

            # excitation
            data = functions.propagate_gradient_pulse_relax(
                pulse_x=self.gp_excitation.data_pulse_x, pulse_y=self.gp_excitation.data_pulse_y,
                grad=self.gp_excitation.data_grad,
                sim_data=data, dt_s=self.gp_excitation.dt_sampling_steps_us * 1e-6
            )
            # if self.settings.visualize:
            #     # save excitation profile snapshot
            #     self.set_magnetization_profile_snap("excitation")

            # delay
            delay_exc_ref1 = functions.matrix_propagation_relaxation_multidim(
                dt_s=self.sequence_timings.get_timing_s("exc_ref1"), sim_data=data
            )
            data = functions.propagate_matrix_mag_vector(delay_exc_ref1, sim_data=data)

            # rf loop
            for rf_idx in tqdm.trange(self.rf_etl, desc="processing sequence, refocusing pulse loop"):
                # propagate pulse
                data = functions.propagate_gradient_pulse_relax(
                    pulse_x=self.gps_refocus[rf_idx].data_pulse_x,
                    pulse_y=self.gps_refocus[rf_idx].data_pulse_y,
                    grad=self.gps_refocus[rf_idx].data_grad,
                    sim_data=data,
                    dt_s=self.gps_refocus[rf_idx].dt_sampling_steps_us * 1e-6
                )
                # if self.settings.visualize:
                #     # save profile snapshot after pulse
                #     self.set_magnetization_profile_snap(snap_name=f"refocus_{rf_idx + 1}_post_pulse")

                # timing from ref to e
                data = functions.propagate_matrix_mag_vector(mat_prop_ref_e_time, sim_data=data)
                # acquisitions
                for idx_e in range(self.num_echoes_per_rf):
                    # take the sum of the contributions of the individual spins at central readout echo time
                    data = functions.sum_sample_acquisition(
                        etl_idx=self.num_echoes_per_rf * rf_idx + idx_e, params=self.params, sim_data=data,
                        acquisition_duration_s=self.params.duration_acquisition * 1e-6
                    )
                    # delay between readouts
                    data = functions.propagate_matrix_mag_vector(mat_prop_e_e_time, sim_data=data)

                # delay to pulse
                data = functions.propagate_matrix_mag_vector(mat_prop_ref_e_time, sim_data=data)
                # fill in original data
                self.data.magnetization_propagation[:, start:end] = data.magnetization_propagation.cpu()
                self.data.signal_mag[:, start:end] = data.signal_mag.cpu()
                self.data.signal_phase[:, start:end] = data.signal_phase.cpu()
                self.data.signal_tensor[:, start:end] = data.signal_tensor.cpu()

                # if self.settings.visualize:
                #     # save excitation profile snapshot
                #     self.set_magnetization_profile_snap(snap_name=f"refocus_{rf_idx + 1}_post_acquisitions")


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

    emc_path = plib.Path(settings.emc_params_file).absolute()
    if emc_path.is_file():
        params = EmcParameters.load(emc_path)
        params.display()
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
