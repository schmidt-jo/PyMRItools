import numpy as np
import logging
import tqdm

from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D
from pymritools.seqprog.core import Kernel, DELAY, ADC, GRAD
from pymritools.seqprog.sequences import Sequence2D, setup_sequence_cli, build

log_module = logging.getLogger(__name__)


class MEGESSE(Sequence2D):
    def __init__(self, config: PulseqConfig, specs: PulseqSystemSpecs, params: PulseqParameters2D):
        # init Base class
        super().__init__(config=config, specs=specs, params=params, create_excitation_kernel=False)

        log_module.info(f"Init MEGESSE Sequence")
        # set number of GRE echoes beside SE
        self.num_gre: int = 2
        # number of readouts per refocusing (symmetrically placed)
        self.num_e_per_rf: int = 1 + 2 * self.num_gre

        # timing
        self.t_delay_exc_ref1: DELAY = DELAY()
        self.t_delay_ref_se: DELAY = DELAY()

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

        # we need to modify the excitation kernel
        # -> symmetric echoes means we want readouts on fid after excitation = full rephasing
        self.block_excitation = Kernel.excitation_slice_sel(
            params=self.params, system=self.system, pulse_file=self.config.pulse_file,
            use_slice_spoiling=False, adjust_ramp_area=0.0
        )
        # we need to modify the excitation kernel -> add phase and read gradients to prep for readouts between
        # excitation and first refocus
        self._mod_excitation()

        # plot files for visualization
        if self.config.visualize:
            self.block_excitation.plot(path=self.path_figs, name="excitation")
            # self.block_refocus_1.plot(path=self.path_figs, name="refocus-1")
            self.block_refocus.plot(path=self.path_figs, name="refocus")
            # self.block_pf_acquisition.plot(path=self.path_figs, name="partial-fourier-acqusisition")
            self.block_acquisition.plot(path=self.path_figs, name="bu-acquisition")
            self.block_acquisition_neg_polarity.plot(path=self.path_figs, name="bd-acquisition")

        # register all slice select kernel pulse gradients
        self.kernel_pulses_slice_select = [self.block_excitation, self.block_refocus]

        # ToDo:
        # as is now all gesse readouts sample the same phase encode lines as the spin echoes.
        # this would allow joint recon of t2 and t2* contrasts independently
        # but we could also benefit even more from joint recon of all echoes and
        # hence switch up the phase encode scheme even further also in between gesse samplings

    def _mod_excitation(self):
        # need to set up a phase gradient for excitation
        phase_grad_area = self.block_refocus.grad_phase.area[0]
        # timing - rf ringdown time can fall into grad
        start_time = self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s
        rephase_time = self.block_excitation.get_duration() - start_time
        # set up longest phase encode
        max_phase_grad_area = self.params.resolution_n_phase / 2 * self.params.delta_k_phase
        # build longest phase gradient
        grad_phase = GRAD.make_trapezoid(
            channel=self.params.phase_dir,
            area=max_phase_grad_area,
            system=self.system,
            delay_s=start_time,
            duration_s=rephase_time
        )
        # need to prephase first gradient echo readout, does not need to be updated. only need half readout grad area
        # we always end with a blip up gradient before first refocus, hence need to figure out with what we do start:
        bu = False if int(self.num_gre % 2) == 0 else True
        block_read = self.block_acquisition if bu else self.block_acquisition_neg_polarity
        # get whole readout area
        grad_area_readout = np.sum(block_read.grad_read.area)
        grad_read_pre = GRAD.make_trapezoid(
            channel=self.params.read_dir,
            area=- 0.5 * grad_area_readout,
            system=self.system,
            delay_s=start_time,
            duration_s=rephase_time
        )
        # get times
        # duration_phase_grad = self.set_on_grad_raster_time(time=grad_phase.get_duration() - grad_phase.t_delay_s)
        # duration_pre_read = self.set_on_grad_raster_time(time=grad_read_pre.get_duration() - grad_read_pre.t_delay_s)
        # duration_re_slice = self.block_excitation.get_duration() - (
        #     self.block_excitation.rf.t_duration_s + self.block_excitation.rf.t_delay_s
        # )
        #
        # # calculate minimum needed time to play out all grads - this should be just a sanity check,
        # # as above grads were already set on the rephase time
        # duration_min = np.max([duration_phase_grad, duration_pre_read, duration_re_slice])
        # if duration_min > duration_re_slice:
        #     msg = ("Excitation Kernel: readout and / or phase gradient take longer than slice spoiling and rephasing. "
        #            "Adopting not yet implemented. Try to use different slice spoiling settings!")
        #     log_module.error(msg)
        #     raise AttributeError(msg)
        # set gradients
        self.block_excitation.grad_read = grad_read_pre
        self.block_excitation.grad_phase = grad_phase

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
        # quick sanity check, acquisitions should be equally long
        if not np.allclose(self.block_acquisition.get_duration(), self.block_acquisition_neg_polarity.get_duration()):
            msg = ("acquisition for different readout directions differ in duration. "
                   "Something went wrong. Check creation of acquisition kernels!")
            log_module.error(msg)
            raise AssertionError(msg)
        # quick sanity check rf for being symmetric
        if not np.allclose(
                self.block_refocus.rf.t_delay_s + self.block_refocus.rf.t_duration_s / 2,
                self.block_refocus.get_duration() / 2
        ):
            msg = ("midpoint of refocusing pulse not in the middle of refocusing kernel. "
                   "Kernel seems to be assymetric. Something went wrong. Check creation of refocusing kernel!")
            log_module.error(msg)
            raise AssertionError(msg)

        # have num_gre + (2 * num_gre + 1) * etl echoes
        # find midpoint of excitation rf
        t_start = self.block_excitation.rf.t_delay_s + self.block_excitation.rf.t_duration_s / 2
        # find time from excitation mid to mid of first acquisition
        t_exc_gre1 = self.block_excitation.get_duration() - t_start + self.block_acquisition.get_duration() / 2
        # find time between readouts - readouts are equal in length -> twice half a readout duration
        t_e2e = self.block_acquisition.get_duration()

        # find time between midpoint of rf refocus block and echo readout (both are symmetric,
        # hence this is time before or after refocusing needed to readout midpoint)
        t_e_ref = self.block_acquisition.get_duration() / 2 + self.block_refocus.get_duration() / 2

        # now we want to calculate minimum echo spacing
        # time from excitation to refocus (including num_gre readouts)
        t_exci_2_ref = t_exc_gre1 + (self.num_gre - 1) * t_e2e + t_e_ref
        t_ref_se = t_e_ref + self.num_gre * t_e2e

        esp = 2 * np.max([t_exci_2_ref, t_ref_se])

        # time to either side between excitation - ref - se needs to be equal, calculate appropriate delays
        if t_exci_2_ref < esp / 2:
            delay_ref_e = 0.0
            self.t_delay_exc_ref1 = DELAY.make_delay(esp / 2 - t_exci_2_ref, system=self.system)
        else:
            self.t_delay_ref_se = DELAY.make_delay(esp / 2 - t_ref_se, system=self.system)
            delay_ref_e = self.t_delay_ref_se.get_duration()

        # write echo times to array, take the first echo readouts
        self.te = [t_exc_gre1]
        for _ in range(self.num_gre - 1):
            self.te.append(self.te[-1] + t_e2e)

        for k in range(0, self.params.etl):
            # if we would need to add an delay (if ref to e is quicker than from excitation to ref),
            # we would do it before the echoes and after the echoes to symmetrize,
            # hence after the last echo we add the time from echo to refocus twice and any delay twice
            # (except for very first refocusing)
            num_delays = 1 if k == 0 else 2
            self.te.append(self.te[-1] + 2 * t_e_ref + delay_ref_e * num_delays)
            for _ in range(2 * self.num_gre):
                self.te.append(self.te[-1] + t_e2e)

        te_print = [f'{1000 * t:.2f}' for t in self.te]
        log_module.info(f"echo times: {te_print} ms")
        log_module.info(f"time excitation mid to refocus mid: "
                        f"{(t_exci_2_ref+self.t_delay_exc_ref1.get_duration())*1e3} ms")

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
        e_types = ["gre"] * self.num_e_per_rf
        e_types[self.num_gre] = "se"

        # we always have an odd number of echoes, hence always starting and ending with bu
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
            self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=0, no_ref_1=True)

            # -- excitation --
            # add block
            self.sequence.add_block(*self.block_excitation.list_events_to_ns())

            # add gradient echo readouts
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
            # we always have odd number of samplings in between refocusing pulses.
            # to just use one refocus kernel, we just make sure that we end with a blip up gre before first refocus
            bu = False if int(self.num_gre % 2) == 0 else True
            for num_readout in range(self.num_gre):
                if bu:
                    # add bu sampling
                    self.sequence.add_block(*aq_block_bu.list_events_to_ns())
                    id_acq = self.id_bu_acq
                else:
                    # add bd sampling
                    self.sequence.add_block(*aq_block_bd.list_events_to_ns())
                    id_acq = self.id_bd_acq
                # toggle after adding
                bu = not bu
                if not no_adc:
                    # write sampling pattern
                    _ = self._write_sampling_pattern_entry(
                        slice_num=self.trueSliceNum[idx_slice],
                        pe_num=int(self.k_pe_indexes[0, idx_pe_n]),
                        echo_num=num_readout,
                        acq_type=id_acq, echo_type="gre"
                    )

            # delay if necessary
            if self.t_delay_exc_ref1.get_duration() > 1e-7:
                self.sequence.add_block(self.t_delay_exc_ref1.to_simple_ns())

            # successive num gre echoes per rf
            for echo_idx in np.arange(self.params.etl):
                # set flip angle from param list
                self._set_fa_and_update_slice_offset(rf_idx=echo_idx, slice_idx=idx_slice)
                # looping through slices per phase encode, set phase encode for ref
                self._set_phase_grad(phase_idx=idx_pe_n, echo_idx=echo_idx, no_ref_1=True)
                # refocus
                self.sequence.add_block(*self.block_refocus.list_events_to_ns())

                # delay if necessary
                if self.t_delay_ref_se.get_duration() > 1e-7:
                    self.sequence.add_block(self.t_delay_ref_se.to_simple_ns())

                self._add_gesse_readouts(
                    idx_pe_loop=idx_pe_n, idx_slice_loop=idx_slice,
                    idx_echo=echo_idx, no_adc=no_adc
                )

                # delay if necessary
                if self.t_delay_ref_se.get_duration() > 1e-7:
                    self.sequence.add_block(self.t_delay_ref_se.to_simple_ns())

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
