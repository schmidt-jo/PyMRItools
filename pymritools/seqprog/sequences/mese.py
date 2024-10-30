import numpy as np
import logging
import tqdm

from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D
from pymritools.seqprog.core import Kernel, DELAY, ADC
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
        self.kernel_pulses_slice_select = {
            "excitation": self.block_excitation, "refocus_1": self.block_refocus_1,
            "refocus": self.block_refocus
        }
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
        t_start = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        t_exci_ref = self.block_refocus_1.rf.t_delay_s + self.block_refocus_1.rf.t_duration_s / 2 + \
                     self.block_excitation.get_duration() - t_start
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
        for idx_slice in np.arange(0, self.params.resolution_slice_num):
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
