import numpy as np
import logging
import tqdm

from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D
from pymritools.seqprog.core import Kernel, DELAY, ADC
from pymritools.seqprog.sequences import Sequence2D, setup_sequence_cli

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

        # sbbs
        self.block_acquisition: Kernel = Kernel.acquisition_fs(
            params=self.params,
            system=self.system
        )
        self.id_acq_se = "fs_acq"

        self.block_refocus, self.phase_enc_time = Kernel.refocus_slice_sel_spoil(
            params=self.params,
            system=self.system,
            pulse_file=self.config.pulse_file,
            pulse_num=1,
            return_pe_time=True,
            read_gradient_to_prephase=self.block_acquisition.grad_read.area / 2
        )

        self.block_refocus_1: Kernel = Kernel.refocus_slice_sel_spoil(
            params=self.params,
            system=self.system,
            pulse_file=self.config.pulse_file,
            pulse_num=0,
            read_gradient_to_prephase=self.block_acquisition.grad_read.area / 2
        )
        # calculate ramp area to slice select upon refocusing 1
        ramp_area_ref_1 = float(self.block_refocus_1.grad_slice.area[0])

        self.block_excitation = Kernel.excitation_slice_sel(
            params=self.params,
            system=self.system,
            pulse_file=self.config.pulse_file,
            adjust_ramp_area=ramp_area_ref_1
        )

        if self.visualize:
            self.block_excitation.plot(path=self.path_figs, name="excitation")
            self.block_refocus_1.plot(path=self.path_figs, name="refocus_1")
            self.block_refocus.plot(path=self.path_figs, name="refocus")
            self.block_acquisition.plot(path=self.path_figs, name="fs_acquisition")

        # register slice select pulse grad kernels
        self.kernel_pulses_slice_select = [self.block_excitation, self.block_refocus_1, self.block_refocus]

    # __ pypsi __
    # sampling + k traj
    def _set_k_trajectories(self):
        # read direction is always fully oversampled, no trajectories to register
        grad_pre_area = float(np.sum(self.block_refocus.grad_read.area) / 2)
        # calculate trajectory for se readout
        self._register_k_trajectory(
            self.block_acquisition.get_k_space_trajectory(
                pre_read_area=grad_pre_area, fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
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
        t_exci_ref = self.block_refocus_1.rf.t_delay_s + self.block_refocus_1.rf.t_duration_s / 2 + \
                     self.block_excitation.get_duration() - self.block_excitation.rf.t_delay_s - \
                     self.block_excitation.rf.t_duration_s / 2
        t_ref_1_adc = self.block_refocus_1.get_duration() - self.block_refocus_1.rf.t_delay_s - \
                      self.block_refocus_1.rf.t_duration_s / 2 + self.block_acquisition.get_duration() / 2
        t_ref_2_adc = self.block_acquisition.get_duration() / 2 + self.block_refocus.get_duration() / 2

        self.params.esp = 2 * np.max([t_exci_ref, t_ref_1_adc, t_ref_2_adc]) * 1e3
        log_module.info(f"\t\t-found minimum ESP: {self.params.esp:.2f} ms")

        if np.abs(t_ref_1_adc - t_ref_2_adc) > 1e-6:
            log_module.error(f"refocus to adc timing different from adc to refocus. Systematic error in seq. creation")
        t_half_esp = self.params.esp * 1e-3 / 2
        # add delays
        if t_exci_ref < t_half_esp:
            self.delay_exci_ref1 = DELAY.make_delay(t_half_esp - t_exci_ref, system=self.system)
            if not self.delay_exci_ref1.check_on_block_raster():
                err = f"exci ref delay not on block raster"
                log_module.error(err)
        if t_ref_1_adc < t_half_esp:
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
            aq_block.adc = ADC
        else:
            aq_block = self.block_acquisition
        for idx_slice in np.arange(-1, self.params.resolution_slice_num):
            # one loop as intro (-1)
            self._set_fa(echo_idx=0, slice_idx=idx_slice, excitation=True)
            self._set_fa(echo_idx=0, slice_idx=idx_slice)
            # looping through slices per phase encode
            self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=0)
            # apply slice offset
            self._apply_slice_offset(idx_slice=idx_slice)

            # excitation
            # add block
            self.sequence.add_block(*self.block_excitation.list_events_to_ns())

            # delay if necessary
            if self.delay_exci_ref1.get_duration() > 1e-7:
                self.sequence.add_block(self.delay_exci_ref1.to_simple_ns())

            # first refocus
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
                self._set_fa(echo_idx=echo_idx, slice_idx=idx_slice)
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

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.params.number_central_lines + self.params.number_outer_lines, desc="phase encodes"
        )
        # one slice loop for introduction
        self._loop_slices(idx_pe_n=0, no_adc=True)
        # counter for number of scan
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            self._loop_slices(idx_pe_n=idx_n)
            if self.navs_on:
                self._loop_navs()

        log_module.info(f"sequence built!")

    def _set_fa(self, echo_idx: int, slice_idx: int, excitation: bool = False):
        if excitation:
            sbb = self.block_excitation
            fa_rad = self.params.excitation_rf_rad_fa
            phase_rad = self.params.excitation_rf_rad_phase
        else:
            if echo_idx == 0:
                sbb = self.block_refocus_1
            else:
                sbb = self.block_refocus
            fa_rad = self.params.refocusing_rf_rad_fa[echo_idx]
            phase_rad = self.params.refocusing_rf_rad_phase[echo_idx]
        flip = sbb.rf.t_duration_s / sbb.rf.signal.shape[0] * np.sum(np.abs(sbb.rf.signal)) * 2 * np.pi
        # slice adaptive fa scaling - we need true slice position here!
        sbb.rf.signal *= fa_rad / flip * self.rf_slice_adaptive_scaling[self.trueSliceNum[slice_idx]]
        sbb.rf.phase_rad = phase_rad

    def _set_phase_grad(self, echo_idx: int, phase_idx: int):
        # phase gradient placement maximum time was calculated for whole area, this is reflected in self.phase_enc_time
        # however this includes ramps. Hence we need to calculate the amplitudes with the ramps in mind
        area_factors = np.array([0.5, 1.0, 0.5])
        # first get the phase index from the sampling scheme
        idx_phase = self.k_pe_indexes[echo_idx, phase_idx]
        # thens et the block we need to change dependent on the echo index
        if echo_idx == 0:
            sbb = self.block_refocus_1
        else:
            sbb = self.block_refocus
            # if we are further in the echo train we also need to rewind the previous phase encode.
            # get the last phase encode index
            last_idx_phase = self.k_pe_indexes[echo_idx - 1, phase_idx]
            # get the effective phase encode time
            t_pe = np.sum(np.diff(sbb.grad_phase.t_array_s[:4]) * area_factors)
            sbb.grad_phase.amplitude[1:3] = self.phase_areas[last_idx_phase] / t_pe
            
        # set the phase encode into the gradient of the refocussing kernel while slice spoil takes place,
        # default phase before read to -, after read to +
        if np.abs(self.phase_areas[idx_phase]) > 1:
            t_pe = np.sum(np.diff(sbb.grad_phase.t_array_s[-4:]) * area_factors)
            sbb.grad_phase.amplitude[-3:-1] = - self.phase_areas[idx_phase] / t_pe
        else:
            sbb.grad_phase.amplitude = np.zeros_like(sbb.grad_phase.amplitude)

    def _set_end_spoil_phase_grad(self):
        factor = np.array([0.5, 1.0, 0.5])

        # get phase moment of last phase encode
        pe_grad_amp = self.block_refocus.grad_phase.amplitude[-2]
        pe_grad_times = self.block_refocus.grad_phase.t_array_s[-4:]
        delta_times = np.diff(pe_grad_times)
        area = np.sum(delta_times * pe_grad_amp * factor)

        # adopt last grad to inverse area
        pe_end_times = self.block_spoil_end.grad_phase.t_array_s[-4:]
        delta_end_times = np.diff(pe_end_times)
        pe_end_amp = area / np.sum(factor * delta_end_times)
        if np.abs(pe_end_amp) > self.system.max_grad:
            err = f"amplitude violation upon last pe grad setting"
            log_module.error(err)
            raise AttributeError(err)
        self.block_spoil_end.grad_phase.amplitude[1:3] = - pe_end_amp


def build(config: PulseqConfig, specs: PulseqSystemSpecs, params: PulseqParameters2D):
    # might be a core function
    # set object
    sequence = MESE(config=config, specs=specs, params=params)
    # build sequence
    sequence.build()
    # plot if set
    # save


def main():
    parser, config, specs, params = setup_sequence_cli("MESE")

    # run prog
    try:
        build(config=config, specs=specs, params=params)
    except Exception as e:
        parser.print_help()
        log_module.exception(e)


if __name__ == '__main__':
    main()
