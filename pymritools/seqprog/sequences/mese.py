import numpy as np
import logging
import tqdm

from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D
from pymritools.seqprog.core import Kernel, DELAY, ADC, events
from pymritools.seqprog.sequences import Sequence2D, setup_sequence_cli, build

log_module = logging.getLogger(__name__)


class MESE(Sequence2D):
    def __init__(self, config: PulseqConfig, specs: PulseqSystemSpecs, params: PulseqParameters2D):
        super().__init__(config=config, specs=specs, params=params)

        # timing
        self.esp: float = params.esp
        self.delay_exci_ref1: DELAY = DELAY()
        self.delay_ref_adc: DELAY = DELAY()
        self.phase_enc_time: float = 0.0
        self.delay_slice: DELAY = DELAY()

        # sbbs - set identifier for acquisition kernel
        self.id_acq_se = "fs_acq"

        if self.visualize:
            self.block_excitation.plot(path=self.path_figs, name="excitation")
            self.block_refocus_1.plot(path=self.path_figs, name="refocus_1")
            self.block_refocus.plot(path=self.path_figs, name="refocus")
            self.block_acquisition.plot(path=self.path_figs, name="fs_acquisition")

        # register slice select pulse grad kernels
        self.kernels_to_save = {
            "excitation": self.block_excitation, "refocus_1": self.block_refocus_1,
            "refocus": self.block_refocus,
            "acq": self.block_acquisition,
        }

    # __ pypsi __
    # sampling + k traj
    def _set_k_trajectories(self):
        # read direction is always fully (over)sampled, no special trajectories to register
        # prephasing is done in refocusing blocks
        grad_pre_area = float(np.sum(self.block_refocus.grad_read.area) / 2)
        # calculate trajectory for se readout
        self._register_k_trajectory(
            self.block_acquisition.get_k_space_trajectory(
                pre_read_area=grad_pre_area,
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_acq_se
        )

    # emc
    def _fill_emc_info(self):
        pass
    #     t_rephase = (self.block_excitation.get_duration() -
    #                  (self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s))
    #     amp_rephase = self.block_excitation.grad_slice.area[-1] / t_rephase
    #     self.interface.emc.gradient_excitation_rephase = self._set_grad_for_emc(amp_rephase)
    #     self.interface.emc.duration_excitation_rephase = t_rephase * 1e6
    #     self.interface.emc.duration_crush = self.phase_enc_time * 1e6
    #     # etl left unchanged

    # __ private __
    def _calculate_min_esp(self):
        # calculate time between midpoints
        t_start = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        t_exci_ref = self.block_refocus_1.rf.t_delay_s + self.block_refocus_1.rf.t_duration_s / 2 + \
                     self.block_excitation.get_duration() - t_start
        t_ref_1_adc = self.block_refocus_1.get_duration() - self.block_refocus_1.rf.t_delay_s - \
                      self.block_refocus_1.rf.t_duration_s / 2 + self.block_acquisition.get_duration() / 2
        t_ref_2_adc = self.block_acquisition.get_duration() / 2 + self.block_refocus.get_duration() / 2
        
        t_exci_ref = np.round(t_exci_ref * 1e12) / 1e12
        t_ref_1_adc = np.round(t_ref_1_adc * 1e12) / 1e12
        t_ref_2_adc = np.round(t_ref_2_adc * 1e12) / 1e12

        self.params.esp = 2 * np.max([t_exci_ref, t_ref_1_adc, t_ref_2_adc]) * 1e3
        self.esp = self.params.esp
        log_module.info(f"\t\t-found minimum ESP: {self.params.esp:.2f} ms")

        if np.abs(t_ref_1_adc - t_ref_2_adc) > 1e-9:
            log_module.error(f"refocus to adc timing different from adc to refocus. Systematic error in seq. creation")
        t_half_esp = np.round(self.params.esp * 1e-3 / 2 * 1e12) / 1e12
        # add delays
        if t_exci_ref < t_half_esp:
            delay = t_half_esp - t_exci_ref
            self.delay_exci_ref1 = DELAY.make_delay(delay, system=self.system)
            if not self.delay_exci_ref1.check_on_block_raster():
                err = f"exci ref delay not on block raster"
                log_module.error(err)
        else:
            self.delay_ref_adc = DELAY.make_delay(t_half_esp - t_ref_1_adc, system=self.system)
            if not self.delay_ref_adc.check_on_block_raster():
                err = f"adc ref delay not on block raster"
                log_module.error(err)
        tes = np.arange(1, self.params.etl + 1) * self.params.esp
        self.te = tes.tolist()

    def _calculate_slice_delay(self):
        # time per echo train
        t_pre_etl = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        t_etl = self.params.etl * self.params.esp * 1e-3  # esp in ms
        t_post_etl = self.block_acquisition.get_duration() / 2 + self.block_spoil_end.get_duration()

        t_total_etl = t_pre_etl + t_etl + t_post_etl
        self._set_slice_delay(t_total_etl=t_total_etl)

    def _build_variant(self):
        log_module.info(f"build -- calculate minimum ESP")
        self._calculate_min_esp()
        log_module.info(f"build -- calculate slice delay")
        self._calculate_slice_delay()

    def _loop_slices(self, idx_pe_n: int, no_adc: bool = False):
        # adc
        if no_adc:
            aq_block = self.block_acquisition.copy()
            aq_block.adc = ADC()
        else:
            aq_block = self.block_acquisition
        for idx_slice in np.arange(0, self.params.resolution_slice_num).astype(int):
            self._set_fa_and_update_slice_offset(rf_idx=0, slice_idx=idx_slice, excitation=True)
            # looping through slices per phase encode
            self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=0)

            # excitation
            # add block
            self.sequence.add_block(*self.block_excitation.list_events_to_ns())

            # delay if necessary
            if self.delay_exci_ref1.get_duration() > 1e-7:
                self.sequence.add_block(self.delay_exci_ref1.to_simple_ns())

            # first refocus
            self._set_fa_and_update_slice_offset(rf_idx=0, slice_idx=idx_slice)
            # add block
            self.sequence.add_block(*self.block_refocus_1.list_events_to_ns())

            # delay if necessary
            if self.delay_ref_adc.get_duration() > 1e-7:
                self.sequence.add_block(self.delay_ref_adc.to_simple_ns())

            # adc
            self.sequence.add_block(*aq_block.list_events_to_ns())
            if not no_adc:
                # write sampling pattern
                _ = self._write_sampling_pattern_entry(
                    slice_num=self.trueSliceNum[idx_slice],
                    pe_num=int(self.k_pe_indexes[0, idx_pe_n]), echo_num=0,
                    acq_type=self.id_acq_se
                )

            # delay if necessary
            if self.delay_ref_adc.get_duration() > 1e-7:
                self.sequence.add_block(self.delay_ref_adc.to_simple_ns())

            # loop
            for echo_idx in np.arange(1, self.params.etl):
                # set fa
                self._set_fa_and_update_slice_offset(rf_idx=echo_idx, slice_idx=idx_slice)
                # set phase
                self._set_phase_grad(echo_idx=echo_idx, phase_idx=idx_pe_n)
                # add block
                self.sequence.add_block(*self.block_refocus.list_events_to_ns())
                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.sequence.add_block(self.delay_ref_adc.to_simple_ns())

                # adc
                self.sequence.add_block(*aq_block.list_events_to_ns())
                if not no_adc:
                    # write sampling pattern
                    _ = self._write_sampling_pattern_entry(
                        slice_num=self.trueSliceNum[idx_slice],
                        pe_num=int(self.k_pe_indexes[echo_idx, idx_pe_n]),
                        echo_num=echo_idx, acq_type=self.id_acq_se
                    )

                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.sequence.add_block(self.delay_ref_adc.to_simple_ns())
            # spoil end
            self._set_end_spoil_phase_grad()
            self.sequence.add_block(*self.block_spoil_end.list_events_to_ns())
            # insert slice delay
            self.sequence.add_block(self.delay_slice.to_simple_ns())

    def _loop_calibration_sequence(self):
        # we first set the calibration object
        self.use_calibration_seq = True
        # we here we want to create a sequence for slice checks
        # we need to move the readout gradient into the z direction, and prephase it for k-space traversing
        # compute the extent (in mm), take adjacent 10 slices right and left
        fov_slice = self.params.resolution_slice_thickness * (
            self.params.resolution_slice_num + (self.params.resolution_slice_num - 1) * self.params.resolution_slice_gap / 100
        )
        # we sample this with ~ 0.5 mm
        resolution_n_slice = int(np.round(2 * fov_slice))
        # set the oversampling to reduce foldover from body
        os = 2
        # set sampling time
        dwell = self.block_acquisition.grad_read.t_flat_time_s / resolution_n_slice / os

        # make adc
        adc = events.ADC.make_adc(
            num_samples=resolution_n_slice * os,
            dwell=dwell,
            system=self.system,
            delay_s=self.system.adc_dead_time
        )
        delta_k = 1e3 / fov_slice

        # calculate amplitude - we use a negative amplitude, s.th. we can use positive pre and rephasers,
        # reducing grad stress for the slice spoiling gradients
        slice_read_grad = events.GRAD.make_trapezoid(
            channel="z", system=self.system, flat_area=-delta_k * resolution_n_slice,
            flat_time=self.block_acquisition.grad_read.t_flat_time_s
        )
        acq_block = Kernel(adc=adc, grad_slice=slice_read_grad)

        # rebuild spoiling gradients, i.e. insert (p)rephasing slice area
        re_area = slice_read_grad.area / 2
        min_re_grad_time = np.sum(np.diff(self.block_refocus.grad_slice.t_array_s[-4:]))

        grad_slice, grad_slice_delay, grad_slice_spoil_re_time = events.GRAD.make_slice_selective(
            pulse_bandwidth_hz=-self.block_refocus_1.rf.bandwidth_hz,
            slice_thickness_m=self.params.refocusing_grad_slice_scale * self.params.resolution_slice_thickness * 1e-3,
            duration_s=self.params.refocusing_duration * 1e-6,
            system=self.system,
            pre_moment=0,
            re_spoil_moment=-self.params.grad_moment_slice_spoiling - re_area,
            t_minimum_re_grad=min_re_grad_time
        )
        
        # adopt slice gradients
        self.block_refocus_1.grad_slice = grad_slice
        self.block_refocus.grad_slice.amplitude[:4] = grad_slice.amplitude[-4:][::-1]
        self.block_refocus.grad_slice.amplitude[-4:] = grad_slice.amplitude[-4:]

        # reset all kernels
        self._set_fa_and_update_slice_offset(
            rf_idx=0, slice_idx=1, excitation=True
        )
        # set fa
        self._set_fa_and_update_slice_offset(rf_idx=0, slice_idx=1)

        # __ build sequence
        # small start delay
        self.sequence_calibration.add_block(DELAY().make_delay(delay_s=1.0).to_simple_ns())
        # use 10 avgs
        for n in range(10):
            # add excitation pulse with mid slice
            self.sequence_calibration.add_block(*self.block_excitation.list_events_to_ns())
            # delay if necessary
            if self.delay_exci_ref1.get_duration() > 1e-7:
                self.sequence_calibration.add_block(self.delay_exci_ref1.to_simple_ns())

            # first refocus
            # want to set a 0 phase encode and read gradient
            self.block_refocus_1.grad_phase.amplitude = np.zeros_like(self.block_refocus_1.grad_phase.amplitude)
            self.block_refocus_1.grad_read.amplitude = np.zeros_like(self.block_refocus_1.grad_read.amplitude)

            self.sequence_calibration.add_block(*self.block_refocus_1.list_events_to_ns())

            # delay if necessary
            if self.delay_ref_adc.get_duration() > 1e-7:
                self.sequence_calibration.add_block(self.delay_ref_adc.to_simple_ns())

            # prephaser included in refocus block
            # self.sequence_calibration.add_block(*slice_read_pre.list_events_to_ns())
            # readout
            self.sequence_calibration.add_block(*acq_block.list_events_to_ns())
            # rephaser included in rephocus block
            # self.sequence_calibration.add_block(*slice_read_post.list_events_to_ns())

            # delay if necessary
            if self.delay_ref_adc.get_duration() > 1e-7:
                self.sequence_calibration.add_block(self.delay_ref_adc.to_simple_ns())

            # want to set a 0 phase encode and read gradient
            self.block_refocus.grad_phase.amplitude = np.zeros_like(self.block_refocus.grad_phase.amplitude)
            self.block_refocus.grad_read.amplitude = np.zeros_like(self.block_refocus.grad_read.amplitude)

            # pulse train loop
            for echo_idx in np.arange(1, self.params.etl):
                # set fa
                self._set_fa_and_update_slice_offset(rf_idx=echo_idx, slice_idx=1)
                # add block
                self.sequence_calibration.add_block(*self.block_refocus.list_events_to_ns())
                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.sequence_calibration.add_block(self.delay_ref_adc.to_simple_ns())

                # prephaser
                # self.sequence_calibration.add_block(*slice_read_pre.list_events_to_ns())
                # readout
                self.sequence_calibration.add_block(*acq_block.list_events_to_ns())
                # rephaser
                # self.sequence_calibration.add_block(*slice_read_post.list_events_to_ns())

                # delay if necessary
                if self.delay_ref_adc.get_duration() > 1e-7:
                    self.sequence_calibration.add_block(self.delay_ref_adc.to_simple_ns())
            # delay in end
            self.sequence_calibration.add_block(DELAY().make_delay(delay_s=8.0).to_simple_ns())


def main():
    parser, config, specs, params = setup_sequence_cli("MESE")
    # setup sequence object
    mese = MESE(config=config, specs=specs, params=params)
    # run prog
    try:
        build(config=config, sequence=mese, name="mese")
    except Exception as e:
        parser.print_help()
        log_module.exception(e)


if __name__ == '__main__':
    main()
