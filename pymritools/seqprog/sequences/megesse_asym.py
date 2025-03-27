import numpy as np
import logging
import tqdm

from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D
from pymritools.seqprog.core import Kernel, DELAY, ADC, GRAD
from pymritools.seqprog.sequences import Sequence2D, setup_sequence_cli, build

log_module = logging.getLogger(__name__)


class MEGESSE(Sequence2D):
    def __init__(self, config: PulseqConfig, specs: PulseqSystemSpecs, params: PulseqParameters2D):
        # init Base class - relax gradient stress
        super().__init__(config=config, specs=specs, params=params, relax_read_grad_stress=True)

        log_module.info(f"Init MEGESSE Sequence")
        # set number of GRE echoes beside SE
        self.num_gre: int = 4
        self.num_e_per_rf: int = 1 + self.num_gre

        # timing
        self.t_delay_exc_ref1: DELAY = DELAY()
        self.t_delay_ref1_se1: DELAY = DELAY()
        self.t_delay_e2e_bubd: DELAY = DELAY()
        self.t_delay_e2e_bdbu: DELAY = DELAY()

        # its possible to redo the sampling scheme with adjusted etl
        # that is change sampling per readout, we would need pe blips between the gesse samplings,
        # hence we leave this for later

        # sbbs
        # for now lets go with a fs readout, takes more time but for proof of concept easier
        # we sample "blip up" and "blip down" in read direction, SE and GRE vary between the acquisitions
        # blip up is standard acquisition set in base class
        # add id
        self.id_bu_acq: str = "bu_fs"

        # save ac region start and end
        # calculate center of k space and indexes for full sampling band
        k_central_phase = round(self.params.resolution_n_phase / 2)
        k_half_central_lines = round(self.params.number_central_lines / 2)
        # set indexes for start and end of full k space center sampling
        self.ac_start = k_central_phase - k_half_central_lines
        self.ac_end = k_central_phase + k_half_central_lines

        # add blip down acquisition
        self.block_acquisition_neg_polarity = Kernel.acquisition_fs(
            params=self.params, system=self.system, invert_grad_read_dir=True, relax_grad_stress=True
        )
        # we want to add phase blips to the negative polarity readout blocks
        # this way we sample bu and bd readouts with the same phase encode,
        # but move on to another phase encode for the next two readouts.
        # Hence not all echoes between consecutive refocusing pulses are subject to the same phase encodes
        # for simplicity we just add a blip by one line assuming the lines are drawn pseudo randomly anyway.
        # though we need simple checks to not run into the AC region or outside k-space
        self._modify_acquisition_block_neg_polarity()

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

    # megesse
    def _check_phase_blip_gradient(self, idx_phase):
        # if phase index inside AC region or at the edge of k-space we want to return false,
        # in order to not set a phase blip, otherwise we do and return true
        if idx_phase >= self.params.resolution_n_phase - 1:
            return False
        if self.ac_start <= idx_phase <= self.ac_end:
            return False
        return True

    def _set_phase_blip_gradient(self, idx_phase):
        if self._check_phase_blip_gradient(idx_phase):
            self._modify_acquisition_block_neg_polarity()
            return idx_phase + 1
        else:
            self._reset_phase_blip_gradient()
            return idx_phase

    def _reset_phase_blip_gradient(self):
        grad_phase = self.block_acquisition_neg_polarity.grad_phase
        self.block_acquisition_neg_polarity.grad_phase.amplitude = np.zeros_like(grad_phase.amplitude)

    def _modify_acquisition_block_neg_polarity(self):
        # we want a phase encode after the readout gradient that moves exactly one line
        # blip gradient is constant
        grad_phase = GRAD.make_trapezoid(
            channel=self.params.phase_dir,
            area=self.params.delta_k_phase,
            system=self.system
        )
        # gradient is appended after blip down read gradient and can start with its ramp
        grad_read = self.block_acquisition_neg_polarity.grad_read
        t_delay = grad_read.t_array_s[-2]
        grad_phase.t_delay_s = t_delay
        self.block_acquisition_neg_polarity.grad_phase = grad_phase

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
            if isinstance(sbb.grad_read.area, float | int):
                sbb.grad_read.area *= -1
            else:
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
        # have num echoes * etl echoes
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
        # sanity check for other block
        t_ref_e_bd = (
                self.block_refocus.get_duration() - (
                self.block_refocus.rf.t_delay_s + self.block_refocus.rf.t_duration_s / 2)
                + self.block_acquisition_neg_polarity.grad_read.get_duration() / 2
        )
        assert np.allclose(t_ref_e1, t_ref_e_bd)
        # echo to echo from bu to bd
        t_e2e_bubd = self.block_acquisition.get_duration() / 2 + self.block_acquisition_neg_polarity.grad_read.get_duration() / 2
        # echo to echo from bd to bu
        t_e2e_bdbu = (
             self.block_acquisition_neg_polarity.get_duration() - self.block_acquisition_neg_polarity.grad_read.get_duration()
        ) / 2 + self.block_acquisition.get_duration() / 2
        # we need to add the e2e time for each additional GRE readout
        # echo time of first se is twice the bigger time of 1) between excitation and first ref
        # 2) between first ref and se and e2e
        esp_1 = 2 * np.max([t_exc_1ref, t_ref_e1, t_e2e_bdbu, t_e2e_bubd])

        # time to either side between excitation - ref - se needs to be equal, calculate appropriate delays
        # we want this to be a multiple of the E2E time, such that we have symmetrical timings throughout the sequence
        if t_exc_1ref < esp_1 / 2:
            delay_exc_ref = esp_1 / 2 - t_exc_1ref
            self.t_delay_exc_ref1 = DELAY.make_delay(delay_exc_ref, system=self.system)
        else:
            delay_exc_ref = 0
        if t_ref_e1 < esp_1 / 2:
            delay_ref_e = esp_1 / 2 - t_ref_e1
            self.t_delay_ref1_se1 = DELAY.make_delay(delay_ref_e, system=self.system)
        else:
            delay_ref_e = 0
        if t_e2e_bubd < esp_1 / 2:
            delay_e2e_bubd = esp_1 / 2 - t_e2e_bubd
            self.t_delay_e2e_bubd = DELAY.make_delay(delay_e2e_bubd, system=self.system)
        else:
            delay_e2e_bubd = 0

        if t_e2e_bdbu < esp_1 / 2:
            delay_e2e_bdbu = esp_1 / 2 - t_e2e_bdbu
            self.t_delay_e2e_bdbu = DELAY.make_delay(delay_e2e_bdbu, system=self.system)
        else:
            delay_e2e_bdbu = 0

        # we get a spacing between every event (exc - ref, ref - e, e - e, e - ref)
        t_delta = esp_1 / 2
        # write echo times to array
        self.te.append(esp_1)
        for e in range(self.num_gre):
            t_delta = t_e2e_bubd if e % 2 == 0 else t_e2e_bdbu
            t_delay = delay_e2e_bubd if e % 2 == 0 else delay_e2e_bdbu
            self.te.append(self.te[-1] + t_delta + t_delay)
        # after this a rf pulse is played out and we can iteratively add the rest of the echoes
        for k in np.arange(self.num_e_per_rf, self.params.etl * self.num_e_per_rf, self.num_e_per_rf):
            # take last echo time (gre sampling after se) need to add time from gre to rf and from rf to gre (equal)
            # if we would need to add an delay (if ref to e is quicker than from excitation to ref),
            # we would do it before the echoes and after the echoes to symmetrize
            self.te.append(self.te[k - 1] + 2 * (t_ref_e1 + delay_ref_e))
            # take this time and add time between gre and se / readout to readout
            for e in range(self.num_gre):
                t_delta = t_e2e_bubd if e % 2 == 0 else t_e2e_bdbu
                t_delay = delay_e2e_bubd if e % 2 == 0 else delay_e2e_bdbu
                self.te.append(self.te[-1] + t_delta + t_delay)
        te_print = [f'{1000 * t:.2f}' for t in self.te]
        log_module.info(f"echo times: {te_print} ms")
        # deliberately set esp weird to catch it upon processing when dealing with megesse style sequence
        self.esp = -1


    def _add_gesse_readouts(self, idx_pe_loop: int, idx_slice_loop: int, idx_echo: int,
                            no_adc: bool = False, phase_blips: bool = True):
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
        idx_phase_last = self.k_pe_indexes[idx_echo, idx_pe_loop]
        idx_phase_new = idx_phase_last
        for num_readout in range(self.num_e_per_rf):
            if int(num_readout % 2) == 0:
                # add bu sampling
                self.sequence.add_block(*aq_block_bu.list_events_to_ns())
                id_acq = self.id_bu_acq
                if phase_blips and self.params.acceleration_factor > 1.1:
                    idx_phase_new = self._set_phase_blip_gradient(idx_phase=idx_phase_last)
                else:
                    # no blip gradient
                    idx_phase_new = idx_phase_last
                    self._reset_phase_blip_gradient()
            else:
                # add bd sampling
                self.sequence.add_block(*aq_block_bd.list_events_to_ns())
                id_acq = self.id_bd_acq
            if not no_adc:
                # write sampling pattern
                _ = self._write_sampling_pattern_entry(
                    slice_num=self.trueSliceNum[idx_slice_loop],
                    pe_num=int(idx_phase_last),
                    echo_num=self.num_e_per_rf * idx_echo + num_readout,
                    acq_type=id_acq, echo_type=e_types[num_readout],
                    echo_type_num=idx_echo
                )
            if num_readout < self.num_e_per_rf - 1:
                # add inter echo delay
                if id_acq == self.id_bu_acq:
                    self.sequence.add_block(self.t_delay_e2e_bubd.to_simple_ns())
                else:
                    self.sequence.add_block(self.t_delay_e2e_bdbu.to_simple_ns())
                    idx_phase_last = idx_phase_new
        add_phase_area_covered = (idx_phase_last - self.k_pe_indexes[idx_echo, idx_pe_loop]) * self.params.delta_k_phase
        # we pre-phased in phase encode direction to some line, with every blip+
        # we need to re-phase less area to get back to 0. hence we want to give the accumulated area as negative
        return -add_phase_area_covered

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
            accum_pe_area = self._add_gesse_readouts(
                idx_pe_loop=idx_pe_n, idx_slice_loop=idx_slice,
                idx_echo=0, no_adc=no_adc, phase_blips=True
            )

            # delay if necessary
            if self.t_delay_ref1_se1.get_duration() > 1e-7:
                self.sequence.add_block(self.t_delay_ref1_se1.to_simple_ns())

            # successive num gre echoes per rf
            for echo_idx in np.arange(1, self.params.etl):
                # set flip angle from param list
                self._set_fa_and_update_slice_offset(rf_idx=echo_idx, slice_idx=idx_slice)
                # looping through slices per phase encode, set phase encode for ref
                self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=echo_idx, add_rephase=accum_pe_area)
                # refocus
                self.sequence.add_block(*self.block_refocus.list_events_to_ns())

                # delay if necessary
                if self.t_delay_ref1_se1.get_duration() > 1e-7:
                    self.sequence.add_block(self.t_delay_ref1_se1.to_simple_ns())

                accum_pe_area = self._add_gesse_readouts(
                    idx_pe_loop=idx_pe_n, idx_slice_loop=idx_slice,
                    idx_echo=echo_idx, no_adc=no_adc, phase_blips=True
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
