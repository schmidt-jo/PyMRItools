import numpy as np
import logging
import tqdm

from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D
from pymritools.seqprog.core import Kernel, DELAY, ADC
from pymritools.seqprog.sequences import Sequence2D, setup_sequence_cli

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
        # # undersampled readout with symmetrical accelerated sidelobes
        # self.block_se_acq, self.acc_factor_us_read = Kernel.acquisition_sym_undersampled(
        #     params=self.params, system=self.system
        # )
        # for now lets go with a fs readout, takes more time but for proof of concept easier
        # we sample "blip up" and "blip down" in read direction, SE and GRE vary between the acquisitions
        self.block_bu_acq = Kernel.acquisition_fs(
            params=self.params, system=self.system
        )
        # add id
        self.id_bu_acq: str = "bu_fs"

        self.block_bd_acq = Kernel.acquisition_fs(
            params=self.params, system=self.system, invert_grad_read_dir=True
        )
        # add id
        self.id_bd_acq: str = "bd_fs"

        # spoiling at end of echo train - modifications
        self._mod_spoiling_end()

        # refocusing
        self.block_refocus_1, _ = Kernel.refocus_slice_sel_spoil(
            params=self.params, system=self.system, pulse_num=0, return_pe_time=True
        )
        # via kernels we can build slice selective blocks of excitation and refocusing
        # if we leave the spoiling gradient of the first refocus (above) we can merge this into the excitation
        # gradient slice refocus gradient. For this we need to now the ramp area of the
        # slice selective refocus 1 gradient in order to account for it. S.th. the slice refocus gradient is
        # equal to the other refocus spoiling gradients used and is composed of: spoiling, refocusing and
        # accounting for ramp area
        ramp_area = np.trapz(
            x=self.block_refocus_1.grad_slice.t_array_s[:2],
            y=self.block_refocus_1.grad_slice.amplitude[:2]
        )
        # excitation pulse
        self.block_excitation = Kernel.excitation_slice_sel(
            params=self.params, system=self.system, adjust_ramp_area=ramp_area
        )

        self.block_refocus, self.t_spoiling_pe = Kernel.refocus_slice_sel_spoil(
            params=self.params, system=self.system, pulse_num=1, return_pe_time=True
        )
        # sanity check
        if np.abs(np.sum(self.block_bu_acq.grad_read.area)) - np.abs(np.sum(self.block_bd_acq.grad_read.area)) > 1e-8:
            err = f"readout areas of echo readouts differ"
            log_module.error(err)
            raise ValueError(err)

        # for the first we need a different gradient rewind to rephase the partial fourier readout k-space travel
        # self.block_refocus_1: Kernel = Kernel.copy(self.block_refocus)
        # self._mod_first_refocus_rewind_0_echo(grad_pre_area=grad_read_pre_area)
        # # need to adapt echo read prewinder and rewinder
        # self._mod_block_prewind_echo_read(self.block_refocus_1)

        self._mod_block_rewind_echo_read(self.block_refocus)
        self._mod_block_prewind_echo_read(self.block_refocus)

        # plot files for visualization
        if self.params.visualize:
            self.block_excitation.plot(path=self.interface.config.output_path, name="excitation")
            self.block_refocus_1.plot(path=self.interface.config.output_path, name="refocus-1")
            self.block_refocus.plot(path=self.interface.config.output_path, name="refocus")
            # self.block_pf_acquisition.plot(path=self.interface.config.output_path, name="partial-fourier-acqusisition")
            self.block_bu_acq.plot(path=self.interface.config.output_path, name="bu-acquisition")
            self.block_bd_acq.plot(path=self.interface.config.output_path, name="bd-acquisition")

        # register all slice select kernel pulse gradients
        self.kernel_pulses_slice_select = [self.block_excitation, self.block_refocus_1, self.block_refocus]

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
            self.block_bu_acq.get_k_space_trajectory(
                pre_read_area=np.sum(self.block_refocus.grad_read.area[-1]),
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_bu_acq
        )
        # calculate trajectory for bd readouts, prephasing is the prephase gre area + whole bu area
        pre_area_bd = np.sum(self.block_refocus.grad_read.area[-1]) + np.sum(self.block_bu_acq.grad_read.area)
        self._register_k_trajectory(
            self.block_bd_acq.get_k_space_trajectory(
                pre_read_area=pre_area_bd,
                fs_grad_area=self.params.resolution_n_read * self.params.delta_k_read
            ),
            identifier=self.id_bd_acq
        )

    # emc
    def _fill_emc_info(self):
        t_rephase = (self.block_excitation.get_duration() -
                     (self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s))
        amp_rephase = self.block_excitation.grad_slice.area[-1] / t_rephase
        self.interface.emc.gradient_excitation_rephase = self._set_grad_for_emc(amp_rephase)
        self.interface.emc.duration_excitation_rephase = t_rephase * 1e6

    def _mod_spoiling_end(self):
        # want to enable complete refocusing of read gradient when spoiling factor -0.5 is chosen in opts
        readout_area = np.trapz(
            x=self.block_bd_acq.grad_read.t_array_s,
            y=self.block_bd_acq.grad_read.amplitude
        )
        spoil_area = self.params.read_grad_spoiling_factor * readout_area
        # now we need to plug in new amplitude into spoiling read gradient
        t_sr = np.sum(
            np.diff(
                self.block_spoil_end.grad_read.t_array_s[-4:]
            ) * np.array([0.5, 1.0, 0.5])
        )
        self.block_spoil_end.grad_read.amplitude[-3:-1] = spoil_area / t_sr

    def _mod_block_prewind_echo_read(self, sbb: Kernel):
        # need to prewind readout echo gradient
        area_read = np.sum(self.block_bu_acq.grad_read.area)
        area_prewind = - 0.5 * area_read
        delta_times_last_grad_part = np.diff(sbb.grad_read.t_array_s[-4:])
        amplitude = area_prewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_times_last_grad_part)
        if np.abs(amplitude) > self.system.max_grad:
            err = f"amplitude violation when prewinding first echo readout gradient"
            log_module.error(err)
            raise ValueError(err)
        sbb.grad_read.amplitude[-3:-1] = amplitude
        sbb.grad_read.area[-1] = area_prewind

    def _mod_block_rewind_echo_read(self, sbb: Kernel):
        # need to rewind readout echo gradient
        if self.num_gre % 2 == 0:
            # even number of GRE readouts after / before SE, we need to rewind the last bu grad
            block_acq = self.block_bu_acq
        else:
            # odd number of GRE readouts after / before SE, we need to rewind the last bd grad
            block_acq = self.block_bd_acq
        area_read = np.sum(block_acq.grad_read.area)
        area_rewind = - 0.5 * area_read
        delta_t_first_grad_part = np.diff(sbb.grad_read.t_array_s[:4])
        amplitude = area_rewind / np.sum(np.array([0.5, 1.0, 0.5]) * delta_t_first_grad_part)
        if np.abs(amplitude) > self.system.max_grad:
            err = f"amplitude violation when prewinding first echo readout gradient"
            log_module.error(err)
            raise ValueError(err)
        sbb.grad_read.amplitude[1:3] = amplitude
        sbb.grad_read.area[0] = area_rewind

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
        t_post_etl = self.block_bd_acq.get_duration() / 2 + self.block_spoil_end.get_duration()
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

        # find time between mid refocus and first and second echo
        t_ref_e1 = (
            self.block_refocus_1.get_duration() - (
                self.block_refocus_1.rf.t_delay_s + self.block_refocus_1.rf.t_duration_s / 2)
            + self.block_bu_acq.get_duration() / 2)
        t_e2e = self.block_bu_acq.get_duration() / 2 + self.block_bd_acq.get_duration() / 2
        # we need to add the e2e time for each additional GRE readout
        # echo time of first se is twice the bigger time of 1) between excitation and first ref
        # 2) between first ref and se
        esp_1 = 2 * np.max([t_exc_1ref, t_ref_e1])

        # time to either side between excitation - ref - se needs to be equal, calculate appropriate delays
        if t_exc_1ref < esp_1 / 2:
            self.t_delay_exc_ref1 = DELAY.make_delay(esp_1 / 2 - t_exc_1ref, system=self.system)
        if t_ref_e1 < esp_1 / 2:
            self.t_delay_ref1_se1 = DELAY.make_delay(esp_1 / 2 - t_ref_e1, system=self.system)

        # write echo times to array
        self.te.append(esp_1)
        for _ in range(self.num_gre):
            self.te.append(self.te[-1] + t_e2e)
        # after this a rf pulse is played out and we can iteratively add the rest of the echoes
        for k in np.arange(self.num_e_per_rf, self.params.etl * self.num_e_per_rf, self.num_e_per_rf):
            # take last echo time (gre sampling after se) need to add time from gre to rf and from rf to gre (equal)
            self.te.append(self.te[k - 1] + 2 * t_ref_e1)
            # take this time and add time between gre and se / readout to readout
            for _ in range(self.num_gre):
                self.te.append(self.te[-1] + t_e2e)
        te_print = [f'{1000 * t:.2f}' for t in self.te]
        log_module.info(f"echo times: {te_print} ms")
        # deliberately set esp weird to catch it upon processing when dealing with vespa/megesse style sequence
        self.esp = -1

    def _set_fa(self, rf_idx: int, slice_idx: int):
        # we take same kernels for different refocusing pulses when going through the sequence
        # want to adopt rf flip angle and phase based on given input parameters via options
        block = self._get_refocus_block_from_echo_idx(rf_idx=rf_idx)
        # calculate flip angle as given
        flip = block.rf.t_duration_s / block.rf.signal.shape[0] * np.sum(np.abs(block.rf.signal)) * 2 * np.pi
        # take flip angle in radiants from options
        fa_rad = self.params.refocusing_rf_rad_fa[rf_idx]
        # take phase as given in options
        phase_rad = self.params.refocusing_rf_rad_phase[rf_idx]
        # slice dep rf scaling
        rf_scaling = self.rf_slice_adaptive_scaling[slice_idx]
        # set block values
        block.rf.signal *= fa_rad / flip * rf_scaling
        block.rf.phase_rad = phase_rad

    def _get_refocus_block_from_echo_idx(self, rf_idx: int) -> Kernel:
        # want to choose the rf based on position in echo train
        if rf_idx == 0:
            # first refocusing is different kernel
            block = self.block_refocus_1
        else:
            # we are on usual gesse echoes, past the first refocus
            block = self.block_refocus
        return block

    def _set_phase_grad(self, echo_idx: int, phase_idx: int, excitation: bool = False):
        # caution we assume trapezoidal phase encode gradients
        area_factors = np.array([0.5, 1.0, 0.5])
        # we get the actual line index from the sampling pattern, dependent on echo number and phase index in the loop
        idx_phase = self.k_pe_indexes[echo_idx, phase_idx]
        # additionally we need the last blocks phase encode for rephasing
        if echo_idx > 0:
            # if we are not on the first readout:
            # we need the last phase encode value to reset before refocusing
            last_idx_phase = self.k_pe_indexes[echo_idx - 1, phase_idx]
        else:
            # we need the phase encode from the 0th echo, as is now it is also encoded like the refocused se readout
            last_idx_phase = self.k_pe_indexes[echo_idx, phase_idx]
        block = self._get_refocus_block_from_echo_idx(rf_idx=echo_idx)
        # if not on excitation we set the re-phase phase encode gradient
        phase_enc_time_pre_pulse = np.sum(np.diff(block.grad_phase.t_array_s[:4]) * area_factors)
        block.grad_phase.amplitude[1:3] = self.phase_areas[last_idx_phase] / phase_enc_time_pre_pulse
        # we get the time of the phase encode after pulse for every event
        phase_enc_time_post_pulse = np.sum(np.diff(block.grad_phase.t_array_s[-4:]) * area_factors)

        # we set the post pulse phase encode gradient that sets up the next readout
        if np.abs(self.phase_areas[idx_phase]) > 1:
            block.grad_phase.amplitude[-3:-1] = - self.phase_areas[idx_phase] / phase_enc_time_post_pulse
        else:
            block.grad_phase.amplitude = np.zeros_like(block.grad_phase.amplitude)

    def _add_gesse_readouts(self, idx_pe_loop: int, idx_slice_loop: int, idx_echo: int, no_adc: bool = False):
        if no_adc:
            # bu readout
            aq_block_bu = self.block_bu_acq.copy()
            aq_block_bu.adc = events.ADC()
            # bd readout
            aq_block_bd = self.block_bd_acq.copy()
            aq_block_bd.adc = events.ADC()
        else:
            aq_block_bu = self.block_bu_acq
            aq_block_bd = self.block_bd_acq
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
                self.pp_seq.add_block(*aq_block_bu.list_events_to_ns())
                id_acq = self.id_bu_acq
            else:
                # add bd sampling
                self.pp_seq.add_block(*aq_block_bd.list_events_to_ns())
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
            # apply slice offset for all kernels
            self._apply_slice_offset(idx_slice=idx_slice)

            # -- excitation --
            # add block
            self.pp_seq.add_block(*self.block_excitation.list_events_to_ns())

            # delay if necessary
            if self.t_delay_exc_ref1.get_duration() > 1e-7:
                self.pp_seq.add_block(self.t_delay_exc_ref1.to_simple_ns())

            # -- first refocus --
            # set flip angle from param list
            self._set_fa(rf_idx=0, slice_idx=idx_slice)
            # looping through slices per phase encode, set phase encode for ref 1
            self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=0)
            # add block
            self.pp_seq.add_block(*self.block_refocus_1.list_events_to_ns())

            # delay if necessary
            if self.t_delay_ref1_se1.get_duration() > 1e-7:
                self.pp_seq.add_block(self.t_delay_ref1_se1.to_simple_ns())

            # add bu and bd samplings
            self._add_gesse_readouts(
                idx_pe_loop=idx_pe_n, idx_slice_loop=idx_slice,
                idx_echo=0, no_adc=no_adc
            )

            # successive num gre echoes per rf
            for echo_idx in np.arange(1, self.params.etl):
                # set flip angle from param list
                self._set_fa(rf_idx=echo_idx, slice_idx=idx_slice)
                # looping through slices per phase encode, set phase encode for ref 1
                self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=echo_idx)
                # refocus
                self.pp_seq.add_block(*self.block_refocus.list_events_to_ns())

                self._add_gesse_readouts(
                    idx_pe_loop=idx_pe_n, idx_slice_loop=idx_slice,
                    idx_echo=echo_idx, no_adc=no_adc
                )

            # set phase encode of final spoiling grad
            self._set_end_spoil_phase_grad()
            # end with spoiling
            self.pp_seq.add_block(*self.block_spoil_end.list_events_to_ns())
            # set slice delay
            self.pp_seq.add_block(self.delay_slice.to_simple_ns())

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

    def _set_end_spoil_phase_grad(self):
        factor = np.array([0.5, 1.0, 0.5])

        # get phase moment of last phase encode
        pe_last_area = np.trapz(
            x=self.block_refocus.grad_phase.t_array_s[-4:],
            y=self.block_refocus.grad_phase.amplitude[-4:]
        )
        # adopt last grad to inverse area
        pe_end_times = self.block_spoil_end.grad_phase.t_array_s[-4:]
        delta_end_times = np.diff(pe_end_times)
        pe_end_amp = pe_last_area / np.sum(factor * delta_end_times)
        if np.abs(pe_end_amp) > self.system.max_grad:
            err = f"amplitude violation upon last pe grad setting"
            log_module.error(err)
            raise AttributeError(err)
        self.block_spoil_end.grad_phase.amplitude[1:3] = - pe_end_amp


if __name__ == '__main__':
    pass
