import numpy as np
import logging
import tqdm

from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D
from pymritools.seqprog.core import Kernel, DELAY, ADC
from pymritools.seqprog.sequences import Sequence2D, setup_sequence_cli, build

log_module = logging.getLogger(__name__)


class MEGESSE(Sequence2D):
    def __init__(self, config: PulseqConfig, specs: PulseqSystemSpecs, params: PulseqParameters2D):
        # init Base class
        super().__init__(config=config, specs=specs, params=params)

        log_module.info(f"Init MEGESSE Sequence")
        # set number of GRE echoes beside SE
        self.num_gre: int = 2
        self.num_e_per_rf: int = 1 + self.num_gre

        # timing
        self.t_delay_exc_ref1: DELAY = DELAY()
        self.t_delay_ref1_se1: DELAY = DELAY()

        # its possible to redo the sampling scheme with adjusted etl
        # that is change sampling per readout, we would need pe blips between the gesse samplings,
        # hence we leave this for later

        # sbbs
        # for now lets go with a fs readout, takes more time but for proof of concept easier
        # we sample "blip up" and "blip down" in read direction, SE and GRE vary between the acquisitions
        # blip up is standard acquisition set in base class
        # add id
        self.id_bu_acq: str = "bu_fs"

        # add blip down acquisition
        self.block_acquisition_neg_polarity = Kernel.acquisition_fs(
            params=self.params, system=self.system, invert_grad_read_dir=True
        )
        # add id
        self.id_bd_acq: str = "bd_fs"

        # sanity check
        if (
                np.abs(np.sum(self.block_acquisition.grad_read.area)) -
                np.abs(np.sum(self.block_acquisition_neg_polarity.grad_read.area))
                > 1e-8
        ):
            err = f"readout areas of echo readouts differ"
            log_module.error(err)
            raise ValueError(err)

        # spoiling at end of echo train - modifications to base class if wanted - depend on last readout polarity
        self._check_and_mod_echo_read_with_last_gre_readout_polarity(sbb=self.block_spoil_end)

        # dependent on the number of gradient echo readouts in the readout train we might need to change the sign
        # of the rewinding gradient lobes. I.e. we prewind the readout gradient moment after the refocusing and
        # to balance the sequence we need to rewind the readout gradient moment before the next refocusing.
        # The number of readouts determines the last readout polarity.
        # First refocusing doesnt need to rewind anything
        self._check_and_mod_echo_read_with_last_gre_readout_polarity(self.block_refocus)

        # plot files for visualization
        if self.config.visualize:
            self.block_excitation.plot(path=self.path_figs, name="excitation")
            self.block_refocus_1.plot(path=self.path_figs, name="refocus-1")
            self.block_refocus.plot(path=self.path_figs, name="refocus")
            # self.block_pf_acquisition.plot(path=self.path_figs, name="partial-fourier-acqusisition")
            self.block_acquisition.plot(path=self.path_figs, name="bu-acquisition")
            self.block_acquisition_neg_polarity.plot(path=self.path_figs, name="bd-acquisition")

        # register all slice select kernel pulse gradients
        self.kernels_to_save = {
            "excitation": self.block_excitation, "refocus_1": self.block_refocus_1,
            "refocus": self.block_refocus,
            "acq_bu": self.block_acquisition, "acq_bd": self.block_acquisition_neg_polarity,
        }

        # ToDo:
        # as is now all gesse readouts sample the same phase encode lines as the spin echoes.
        # this would allow joint recon of t2 and t2* contrasts independently
        # but we could also benefit even more from joint recon of all echoes and
        # hence switch up the phase encode scheme even further also in between gesse samplings

    # __ pypsi __
    # sampling + k-traj
    def _set_k_trajectories(self):
        # get all read - k - trajectories
        # calculate trajectory for gre readout, prephasing area = to refocus block read area half
        self._register_k_trajectory(
            self.block_acquisition.get_k_space_trajectory(
                pre_read_area=np.sum(self.block_refocus.grad_read.area[-1]),
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_bu_acq
        )
        # calculate trajectory for bd readouts, prephasing is the prephase gre area + whole bu area
        pre_area_bd = np.sum(self.block_refocus.grad_read.area[-1]) + np.sum(self.block_acquisition.grad_read.area)
        self._register_k_trajectory(
            self.block_acquisition_neg_polarity.get_k_space_trajectory(
                pre_read_area=pre_area_bd,
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_bd_acq
        )

    # emc
    def _fill_emc_info(self):
        # t_rephase = (self.block_excitation.get_duration() -
        #              (self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s))
        # amp_rephase = self.block_excitation.grad_slice.area[-1] / t_rephase
        # self.interface.emc.gradient_excitation_rephase = self._set_grad_for_emc(amp_rephase)
        # self.interface.emc.duration_excitation_rephase = t_rephase * 1e6
        pass

    # def _mod_spoiling_end(self):
    #     # want to enable complete refocusing of read gradient when spoiling factor -0.5 is chosen in opts
    #     # get correct last gradient
    #     if self.num_gre % 2 == 0:
    #         # even number of GRE readouts after / before SE, the last read gradient is bu grad
    #         block_acq = self.block_bu_acq
    #     else:
    #         # odd number of GRE readouts after / before SE, the last readgradient is bd grad
    #         block_acq = self.block_bd_acq
    #     readout_area = np.trapezoid(
    #         x=block_acq.grad_read.t_array_s,
    #         y=block_acq.grad_read.amplitude
    #     )
    #     spoil_area = self.params.read_grad_spoiling_factor * readout_area
    #     # now we need to plug in new amplitude into spoiling read gradient
    #     t_sr = np.sum(
    #         np.diff(
    #             self.block_spoil_end.grad_read.t_array_s[-4:]
    #         ) * np.array([0.5, 1.0, 0.5])
    #     )
    #     self.block_spoil_end.grad_read.amplitude[-3:-1] = spoil_area / t_sr

    # def _mod_block_prewind_echo_read(self, sbb: Kernel):
    #     # need to prewind readout echo gradient
    #     area_read = np.sum(self.block_bu_acq.grad_read.area)
    #     area_prewind = - 0.5 * area_read
    #     delta_times_last_grad_part = np.diff(sbb.grad_read.t_array_s[-4:])
    #     amplitude = area_prewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_times_last_grad_part)
    #     if np.abs(amplitude) > self.system.max_grad:
    #         err = f"amplitude violation when prewinding first echo readout gradient"
    #         log_module.error(err)
    #         raise ValueError(err)
    #     sbb.grad_read.amplitude[-3:-1] = amplitude
    #     sbb.grad_read.area[-1] = area_prewind

    def _check_and_mod_echo_read_with_last_gre_readout_polarity(self, sbb: Kernel):
        # need to rewind readout echo gradient
        if self.num_gre % 2 == 0:
            # even number of GRE readouts after / before SE, we need to rewind the last bu grad
            block_acq = self.block_acquisition
        else:
            # odd number of GRE readouts after / before SE, we need to rewind the last bd grad
            block_acq = self.block_acquisition_neg_polarity
        # get polarity
        last_readout_polarity = np.sign(np.sum(block_acq.grad_read.area))
        area = sbb.grad_read.area
        if isinstance(area, float | int):
            area = [area]
        if not np.sign(area[-1]) == - last_readout_polarity:
            # need to flip signs
            # assumes trapezoidal gradients - only takes first trapezoid on longer kernels
            sbb.grad_read.amplitude[:4] *= -1
            sbb.grad_read.area[0] *= -1
            # area_read = np.sum(block_acq.grad_read.area)
            # area_rewind = - 0.5 * area_read


    def _build_variant(self):
        log_module.info(f"build -- calculate minimum ESP")
        self._calculate_echo_timings()
        log_module.info(f"build -- calculate slice delay")
        self._calculate_slice_delay()

    def _calculate_slice_delay(self):
        # time per echo train
        # time to mid excitation
        t_pre_etl = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        # time of etl
        t_etl = self.te[-1]
        # time from mid last gre til end
        t_post_etl = self.block_acquisition_neg_polarity.get_duration() / 2 + self.block_spoil_end.get_duration()
        # total echo train length
        t_total_etl = (t_pre_etl + t_etl + t_post_etl)
        self._set_slice_delay(t_total_etl=t_total_etl)

    def _calculate_echo_timings(self):
        # have 2 * etl echoes
        # find midpoint of rf
        t_start = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        # find time between exc and mid first refocus (not symmetrical)
        t_exc_1ref = (self.block_excitation.get_duration() - t_start + self.block_refocus_1.rf.t_delay_s +
                      self.block_refocus_1.rf.t_duration_s / 2)

        # find time between mid refocus to first and to second echo
        t_ref_e1 = (
                self.block_refocus_1.get_duration() - (
                self.block_refocus_1.rf.t_delay_s + self.block_refocus_1.rf.t_duration_s / 2)
                + self.block_acquisition.get_duration() / 2)
        t_e2e = self.block_acquisition.get_duration() / 2 + self.block_acquisition_neg_polarity.get_duration() / 2
        # we need to add the e2e time for each additional GRE readout
        # echo time of first se is twice the bigger time of 1) between excitation and first ref
        # 2) between first ref and se
        esp_1 = 2 * np.max([t_exc_1ref, t_ref_e1])

        # time to either side between excitation - ref - se needs to be equal, calculate appropriate delays
        if t_exc_1ref < esp_1 / 2:
            delay_ref_e = 0.0
            self.t_delay_exc_ref1 = DELAY.make_delay(esp_1 / 2 - t_exc_1ref, system=self.system)
        else:
            delay_ref_e = esp_1 / 2 - t_ref_e1
            self.t_delay_ref1_se1 = DELAY.make_delay(delay_ref_e, system=self.system)

        # write echo times to array
        self.te.append(esp_1)
        for _ in range(self.num_gre):
            self.te.append(self.te[-1] + t_e2e)
        # after this a rf pulse is played out and we can iteratively add the rest of the echoes
        for k in np.arange(self.num_e_per_rf, self.params.etl * self.num_e_per_rf, self.num_e_per_rf):
            # take last echo time (gre sampling after se) need to add time from gre to rf and from rf to gre (equal)
            # if we would need to add an delay (if ref to e is quicker than from excitation to ref),
            # we would do it before the echoes and after the echoes to symmetrize
            self.te.append(self.te[k - 1] + 2 * (t_ref_e1 + delay_ref_e))
            # take this time and add time between gre and se / readout to readout
            for _ in range(self.num_gre):
                self.te.append(self.te[-1] + t_e2e)
        te_print = [f'{1000 * t:.2f}' for t in self.te]
        log_module.info(f"echo times: {te_print} ms")
        # deliberately set esp weird to catch it upon processing when dealing with megesse style sequence
        self.esp = -1


    def _add_gesse_readouts(self, idx_pe_loop: int, idx_slice_loop: int, idx_echo: int, no_adc: bool = False):
        if no_adc:
            # bu readout
            aq_block_bu = self.block_acquisition.copy()
            aq_block_bu.adc = ADC()
            # bd readout
            aq_block_bd = self.block_acquisition_neg_polarity.copy()
            aq_block_bd.adc = ADC()
        else:
            aq_block_bu = self.block_acquisition
            aq_block_bd = self.block_acquisition_neg_polarity
        # phase encodes are set up to be equal per echo
        # set echo type list
        e_types = ["gre"] * self.num_gre
        if int(idx_echo % 2) == 0:
            e_types.insert(0, "se")
        else:
            e_types.append("se")

        for num_readout in range(self.num_e_per_rf):
            if int(num_readout % 2) == 0:
                # add bu sampling
                self.sequence.add_block(*aq_block_bu.list_events_to_ns())
                id_acq = self.id_bu_acq
            else:
                # add bd sampling
                self.sequence.add_block(*aq_block_bd.list_events_to_ns())
                id_acq = self.id_bd_acq
            if not no_adc:
                # write sampling pattern
                _ = self._write_sampling_pattern_entry(
                    slice_num=self.trueSliceNum[idx_slice_loop],
                    pe_num=int(self.k_pe_indexes[idx_echo, idx_pe_loop]),
                    echo_num=self.num_e_per_rf * idx_echo + num_readout,
                    acq_type=id_acq, echo_type=e_types[num_readout],
                    echo_type_num=idx_echo
                )

    def _loop_slices(self, idx_pe_n: int, no_adc: bool = False):
        for idx_slice in range(self.params.resolution_slice_num):
            # caution we need to set the fa before applying slice offset.
            # otherwise the phase parameter might not be correct for phase offset update
            self._set_fa_and_update_slice_offset(rf_idx=0, slice_idx=idx_slice, excitation=True)
            # looping through slices per phase encode
            self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=0)

            # -- excitation --
            # add block
            self.sequence.add_block(*self.block_excitation.list_events_to_ns())

            # delay if necessary
            if self.t_delay_exc_ref1.get_duration() > 1e-7:
                self.sequence.add_block(self.t_delay_exc_ref1.to_simple_ns())

            # -- first refocus --
            # set flip angle from param list
            self._set_fa_and_update_slice_offset(rf_idx=0, slice_idx=idx_slice)

            # looping through slices per phase encode, set phase encode for ref 1
            self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=0)
            # add block
            self.sequence.add_block(*self.block_refocus_1.list_events_to_ns())

            # delay if necessary
            if self.t_delay_ref1_se1.get_duration() > 1e-7:
                self.sequence.add_block(self.t_delay_ref1_se1.to_simple_ns())

            # add bu and bd samplings
            self._add_gesse_readouts(
                idx_pe_loop=idx_pe_n, idx_slice_loop=idx_slice,
                idx_echo=0, no_adc=no_adc
            )

            # delay if necessary
            if self.t_delay_ref1_se1.get_duration() > 1e-7:
                self.sequence.add_block(self.t_delay_ref1_se1.to_simple_ns())

            # successive num gre echoes per rf
            for echo_idx in np.arange(1, self.params.etl):
                # set flip angle from param list
                self._set_fa_and_update_slice_offset(rf_idx=echo_idx, slice_idx=idx_slice)
                # looping through slices per phase encode, set phase encode for ref
                self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=echo_idx)
                # refocus
                self.sequence.add_block(*self.block_refocus.list_events_to_ns())

                # delay if necessary
                if self.t_delay_ref1_se1.get_duration() > 1e-7:
                    self.sequence.add_block(self.t_delay_ref1_se1.to_simple_ns())

                self._add_gesse_readouts(
                    idx_pe_loop=idx_pe_n, idx_slice_loop=idx_slice,
                    idx_echo=echo_idx, no_adc=no_adc
                )

                # delay if necessary
                if self.t_delay_ref1_se1.get_duration() > 1e-7:
                    self.sequence.add_block(self.t_delay_ref1_se1.to_simple_ns())

            # set phase encode of final spoiling grad
            self._set_end_spoil_phase_grad()
            # end with spoiling
            self.sequence.add_block(*self.block_spoil_end.list_events_to_ns())
            # set slice delay
            self.sequence.add_block(self.delay_slice.to_simple_ns())

    def _loop_lines(self):
        # through phase encodes
        line_bar = tqdm.trange(
            self.params.number_central_lines + self.params.number_outer_lines, desc="phase encodes"
        )
        # one loop for introduction and settling in, no adcs
        self._loop_slices(idx_pe_n=0, no_adc=True)
        for idx_n in line_bar:  # We have N phase encodes for all ETL contrasts
            self._loop_slices(idx_pe_n=idx_n)
            if self.navs_on:
                self._loop_navs()


def main():
    parser, config, specs, params = setup_sequence_cli("MEGESSE")
    # setup sequence object
    megesse = MEGESSE(config=config, specs=specs, params=params)
    # run prog
    try:
        build(config=config, sequence=megesse, name="megesse")
    except Exception as e:
        parser.print_help()
        log_module.exception(e)


if __name__ == '__main__':
    main()
