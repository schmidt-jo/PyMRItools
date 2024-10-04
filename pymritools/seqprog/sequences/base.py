from pymritools.seqprog.core import kernels, events
from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters
from pymritools.config import setup_program_logging, setup_parser
import numpy as np
import logging
from pypulseq import Opts, Sequence
import pathlib as plib
import abc
import json

log_module = logging.getLogger(__name__)


class Sequence2D(abc.ABC):
    """
    Base class for 2D sequences.
    initializing some basic configuration shared for all 2d sequences, such as:
    - slice sampling patterns (interleaved, sequential)
    - slice adaptive scaling of rf (based on in vivo sampling scans)
    - k-space sampling patterns (fs/pf in read, AC region + compementary or arbitrary encodes in phase direction
    - random default generator seed 0, eg. for reproducing sampling patterns
    - checking interface setting
    - navigator toggling (navigators are set outside of the slice slab to not interfere with slice selection / TR
        of slices to measure, they are epi style readouts with lower resolution and can be switched on here)

    """
    def __init__(self, config: PulseqConfig, specs: PulseqSystemSpecs, params: PulseqParameters):

        self.config: PulseqConfig = config
        self.visualize: bool = config.visualize
        # set paths and flags
        path_out = plib.Path(config.out_path).absolute()
        log_module.info(f"Setting output path: {path_out.as_posix()}")
        if not path_out.exists():
            log_module.info("mkdir".ljust(20))
            path_out.mkdir(parents=True, exist_ok=True)
        if self.visualize:
            path_figs = path_out.joinpath("figs")
            log_module.info(f"Setting figure path: {path_figs.as_posix()}")
            if not path_figs.exists():
                log_module.info("mkdir".ljust(20))
                path_figs.mkdir(parents=True, exist_ok=True)
            self.path_figs = path_figs
        else:
            self.path_figs = None

        # set pulseq parameter configuration
        self.params: PulseqParameters = params

        # get system specs
        self.specs: PulseqSystemSpecs = specs
        self.system: Opts = self._set_system_specs()

        # set pypulseq sequence as var -> thats actually whats been build throughout the code and
        # later shipped to a .seq file
        self.sequence: Sequence = Sequence(system=self.system)

        # track echoes
        self.te: list = []

        self.phase_areas: np.ndarray = (- np.arange(
            self.params.resolution_n_phase
        ) + self.params.resolution_n_phase / 2) * self.params.delta_k_phase
        # slice loop
        self.num_slices = self.params.resolution_slice_num
        self.z = np.zeros((2, int(np.ceil(self.num_slices / 2))))

        self.trueSliceNum = np.zeros(self.num_slices, dtype=int)
        # k space
        self.k_pe_indexes: np.ndarray = np.zeros(
            (self.params.etl,
             self.params.number_central_lines + self.params.number_outer_lines
             ),
            dtype=int
        )
        # set exciation and refocusing at least to be present
        self.block_excitation: kernels.Kernel = NotImplemented
        self.block_refocus: kernels.Kernel = NotImplemented
        # set up spoling at end of echo train
        self.block_spoil_end: kernels.Kernel = kernels.Kernel.spoil_all_grads(
            params=self.params, system=self.system
        )
        # register all pulses that need slice select
        self.kernel_pulses_slice_select: list = []

        self._sampling_pattern_constr: list = []
        self.block_excitation: kernels.Kernel = kernels.Kernel()

        # use random state for reproducibility of eg sampling patterns
        # (for that matter any random or semi random sampling used)
        self.prng = np.random.RandomState(0)

        # to check interface set
        self.sampling_pattern_set: bool = False
        self.k_trajectory_set: bool = False
        self.recon_params: bool = False

        # count adcs to track adcs for recon
        self.scan_idx: int = 0
        self.rf_slice_adaptive_scaling: np.ndarray = np.ones(self.params.resolution_slice_num)

        if self.params.rf_adapt_z:
            # set slice adaptive fa scaling, we want to make up for suboptimal FA performance in inferior slices
            # before adapting a ptx scheme we could just try to account for the overall RF intensity decrease
            # by adaptively scaling the RF depending on slice position.
            # This probably wont fix the 2d profile with very bad saturation at temporal rois,
            # but could slightly make up for it.
            # At the expense of increased SAR and possibly central brightening in inferior slices
            # the overall decrease roughly follows a characteristic resembled by part of a sin function
            slice_intensity_profile = np.sin(
                np.linspace(0.9 * np.pi / 4, np.pi / 2, self.params.resolution_slice_num)
                )
            # since we want to make up for this intensity decrease towards lower slices we invert this profile
            self.rf_slice_adaptive_scaling = 1 / slice_intensity_profile

        # navigators
        self.navs_on: bool =self.params.use_navs
        self.nav_num: int = 0
        self.nav_t_total: float = 0.0
        # for now we fix the navigator resolution at 5 times coarser than the chosen resolution
        # of scan read direction. Will be different from scan if not isotropic in plane is used
        self.nav_resolution_factor: int = 5
        if self.navs_on:
            self._set_navigators()

    # __ public __
    # create
    # @classmethod
    # def from_cli(cls, args: PulseqConfig):
    #     # create class instance
    #     pypsi_params = pypsi.Params()
    #     # load different part of cli arguments
    #     loads = [args.i, args.s]
    #     msg = ["sequence configuration", "system specifications"]
    #     att = ["pypulseq", "specs"]
    #     for l_idx in range(len(loads)):
    #         # make plib Path
    #         l_file = plib.Path(loads[l_idx]).absolute()
    #         # check files are provided
    #         if not l_file.is_file():
    #             if l_idx == 0:
    #                 err = f"A {msg[l_idx]} file needs to be provided and {l_file} was not found to be a valid file."
    #                 log_module.error(err)
    #                 raise FileNotFoundError(err)
    #             else:
    #                 warn = f"A {msg[l_idx]} file needs to be provided and {l_file} was not found to be a valid file." \
    #                        f" Falling back to defaults! Check carefully!"
    #                 log_module.warning(warn)
    #         log_module.info(f"loading {msg[l_idx]}: {l_file.as_posix()}")
    #         # set attributes
    #         pypsi_params.__setattr__(att[l_idx], pypsi_params.__getattribute__(att[l_idx]).load(l_file))
    #         if l_idx == 0:
    #             # post stats
    #             log_module.info(f"Bandwidth: {pypsi_params.pypulseq.bandwidth:.3f} Hz/px; "
    #                             f"Readout time: {pypsi_params.pypulseq.acquisition_time * 1e3:.1f} ms; "
    #                             f"DwellTime: {pypsi_params.pypulseq.dwell * 1e6:.1f} us; "
    #                             f"Number of Freq Encodes: {pypsi_params.pypulseq.resolution_n_read}")
    #             _ = pypsi_params.pypulseq.get_voxel_size(write_log=True)
    #
    #     if args.o:
    #         # set output path
    #         o_path = plib.Path(args.o).absolute()
    #         if o_path.suffixes:
    #             md = o_path.parent
    #         else:
    #             md = o_path
    #         # check if exist
    #         md.mkdir(parents=True, exist_ok=True)
    #     else:
    #         # use input path
    #         o_path = plib.Path(args.i).absolute()
    #         if o_path.suffixes:
    #             o_path = o_path.parent
    #         log_module.info(f"no output path specified, using same as input: {o_path}")
    #
    #     pypsi_params.config.output_path = o_path
    #
    #     # overwrite extra arguments if not default_config
    #     d_extra = {
    #         "vv": "version",
    #         "r": "report",
    #         "v": "visualize",
    #         "n": "name",
    #         "t": "type"
    #     }
    #     def_conf = options.Config()
    #     for key, val in d_extra.items():
    #         if def_conf.__getattribute__(key) != args.__getattribute__(key):
    #             pypsi_params.pypulseq.__setattr__(val, args.__getattribute__(key))
    #     pypsi_params.display_sequence_configuration()
    #     return cls(pypsi_params=pypsi_params)

    # get
    def get_pypulseq_seq(self):
        return self.sequence

    def get_z(self):
        # get slice extend
        return self.z

    # writes
    def write_seq(self, name: str = ""):
        file_name = plib.Path(self.config.out_path).absolute()
        if not name:
            name = f"MESE_{self.config.version}"
        name = f"pyp_seq_{name}"
        save_file = file_name.joinpath(name).with_suffix(".seq")
        log_module.info(f"writing file: {save_file.as_posix()}")
        self._check_interface_set()
        self.set_pyp_definitions()
        self.sequence.write(save_file.as_posix())

    def write_pypsi(self, name: str = ""):
        path = plib.Path(self.interface.config.output_path).absolute()
        if not name:
            name = f"{self.params.name}_{self.params.version}"
        name = f"pypsi_{name}"
        save_file = path.joinpath(name).with_suffix(".pkl")
        log_module.info(f"writing file: {save_file.as_posix()}")
        self._check_interface_set()
        # write
        self.interface.save(save_file.as_posix().__str__())

        name = f"z-adapt-rf_{name}"
        save_file = path.joinpath(name).with_suffix(".json")
        log_module.info(f"writing file: {save_file.as_posix()}")
        j_dict = {
            "rf_scaling_z": self.rf_slice_adaptive_scaling.tolist(),
            "z_slice_idx": np.arange(self.params.resolution_slice_num).tolist()}
        with open(save_file.as_posix(), "w") as j_file:
            json.dump(j_dict, j_file, indent=2)

    def set_pyp_definitions(self):
        self.sequence.set_definition(
            "FOV",
            [*self.params.get_fov()]
        )
        self.sequence.set_definition(
            "Name",
            f"mese_{self.config.version}".replace(".", "p")
        )
        self.sequence.set_definition(
            "AdcRasterTime",
            1e-07
        )
        self.sequence.set_definition(
            "GradientRasterTime",
            self.specs.grad_raster_time
        )
        self.sequence.set_definition(
            "RadiofrequencyRasterTime",
            self.specs.rf_raster_time
        )

    def simulate_grad_moments(self, t_end_ms: int, dt_steps_us: int):
        log_module.info(f"simulating gradient moments")
        # build axis of length TR in steps of us
        ax = np.arange(t_end_ms * 1e3 / dt_steps_us)
        t = 0
        # get gradient shapes
        grads = np.zeros((4, ax.shape[0]))  # grads [read, phase, slice, adc]
        # get seq data until defined length
        block_times = np.cumsum(self.sequence.block_durations)
        end_id = np.where(block_times >= t_end_ms * 1e-3)[0][0]
        for block_counter in range(end_id):
            block = self.sequence.get_block(block_counter + 1)
            if getattr(block, "adc", None) is not None:  # ADC
                b_adc = block.adc
                # From Pulseq: According to the information from Klaus Scheffler and indirectly from Siemens this
                # is the present convention - the samples are shifted by 0.5 dwell
                t_start = t + int(1e6 * b_adc.delay / dt_steps_us)
                t_end = t_start + int(1e6 * b_adc.num_samples * b_adc.dwell / dt_steps_us)
                grads[3, t_start:t_end] = 1

            grad_channels = ["gx", "gy", "gz"]
            for x in range(len(grad_channels)):  # Gradients
                if getattr(block, grad_channels[x], None) is not None:
                    grad = getattr(block, grad_channels[x])
                    t_start = t + int(1e6 * grad.delay / dt_steps_us)
                    t_end = t_start + int(1e6 * grad.shape_dur / dt_steps_us)
                    grad_shape = np.interp(np.arange(t_end - t_start), 1e6 * grad.tt / dt_steps_us, grad.waveform)
                    grads[x, t_start:t_end] = grad_shape

            t += int(1e6 * self.sequence.block_durations[block_counter] / dt_steps_us)

        # want to get the moments, basically just cumsum over the grads, multiplied by delta t = 5us
        grad_moments = np.copy(grads)
        grad_moments[:3] = np.cumsum(grads[:3], axis=1) * dt_steps_us * 1e-6
        # do lazy maximization to 2 for visual purpose, we are only interested in visualizing the drift
        grad_moments[:3] = 2 * grad_moments[:3] / np.max(np.abs(grad_moments[:3]), axis=1, keepdims=True)
        # want to plot the moments
        # if self.params.visualize:
        #     self._plot_grad_moments(grad_moments, dt_in_us=dt_steps_us)

    def build(self):
        log_module.info(f"__Build Sequence__")
        log_module.info(f"build -- calculate total scan time")
        self._calculate_scan_time()
        log_module.info(f"build -- set up k-space")
        self._set_k_space()
        log_module.info(f"build -- set up slices")
        self._set_delta_slices()
        log_module.info(f"build variant specifics")
        self._build_variant()
        log_module.info(f"build -- loop lines")
        # prescan for noise correlation
        self._noise_pre_scan()
        self._loop_lines()
        log_module.info(f"set recon info data")
        # sampling + k traj
        self._set_k_trajectories()  # raises error if not implemented
        self._write_sampling_pattern()
        # recon
        self._set_recon_parameters_img()
        self._set_nav_parameters()  # raises error if not implemented
        # emc
        self._set_emc_parameters()  # raises error if not implemented
        # pulse
        self._set_pulse_info()

    # __ private __
    @abc.abstractmethod
    def _build_variant(self):
        # to be defined for each sequence variant
        pass

    @abc.abstractmethod
    def _loop_lines(self):
        # to be implemented for each variant, looping through the phase encodes
        pass

    def _noise_pre_scan(self):
        # make delay
        post_delay = events.DELAY.make_delay(delay_s=0.1, system=self.system)
        # build adc block
        acq = events.ADC.make_adc(system=self.system, num_samples=1000, dwell=self.params.dwell)
        # use 2 noise scans
        for k in range(2):
            # add to sequence
            self.sequence.add_block(acq.to_simple_ns())
            # write as sampling entry
            self._write_sampling_pattern_entry(
                slice_num=0, pe_num=0, echo_num=k, echo_type="noise_scan", acq_type="noise_scan"
            )
            self.sequence.add_block(post_delay.to_simple_ns())

    # caution: this is closely tied to the pypsi module and changes in either might affect the other!
    def _check_interface_set(self):
        if any([not state for state in [self.k_trajectory_set, self.recon_params_set, self.sampling_pattern_set]]):
            warn = f"pypsi interface might not have been set:" \
                   f" Sampling Pattern ({self.sampling_pattern_set}), K-Trajectory ({self.k_trajectory_set})," \
                   f" Recon Parameters ({self.recon_params_set})"
            log_module.warning(warn)
        for acq_type in self.interface.sampling_k_traj.sampling_pattern["acq_type"].unique():
            if acq_type not in self.interface.sampling_k_traj.k_trajectories["acquisition"].unique():
                warn = f"acquisition registered in sampling pattern: " \
                       f"{acq_type} not registered in k-trajectory - types: " \
                       f"{self.interface.sampling_k_traj.k_trajectories['acquisition'].unique()}"
                log_module.warning(warn)

    # sampling & k - space
    def _write_sampling_pattern_entry(self, slice_num: int, pe_num: int, echo_num: int,
                                      acq_type: str = "", echo_type: str = "", echo_type_num: int = -1,
                                      nav_acq: bool = False):
        log_module.debug(f"set pypsi sampling pattern")
        self.sampling_pattern_set = True
        # save to list
        self._sampling_pattern_constr.append({
            "scan_num": self.scan_idx, "slice_num": slice_num, "pe_num": pe_num, "acq_type": acq_type,
            "echo_num": echo_num, "echo_type": echo_type, "echo_type_num": echo_type_num,
            "nav_acq": nav_acq
        })
        self.scan_idx += 1
        return echo_type_num + 1

    def _write_sampling_pattern(self):
        self.interface.sampling_k_traj.sampling_pattern_from_list(sp_list=self._sampling_pattern_constr)

    @abc.abstractmethod
    def _set_k_trajectories(self):
        log_module.debug(f"set pypsi k-traj")
        # to be implemented by sequence variants
        pass

    def _register_k_trajectory(self, trajectory: np.ndarray, identifier: str):
        log_module.debug(f"pypsi: register k - trajectory ({identifier}) in interface")
        self.k_trajectory_set = True
        # build shorthand
        self.interface.sampling_k_traj.register_trajectory(
            trajectory=trajectory, identifier=identifier
        )

    # recon info
    def _set_recon_parameters_img(self):
        log_module.debug(f"set pypsi recon")
        self.interface.recon.set_recon_params(
            img_n_read=self.params.resolution_n_read, img_n_phase=self.params.resolution_n_phase,
            img_n_slice=self.params.resolution_slice_num,
            img_resolution_read=self.params.resolution_voxel_size_read,
            img_resolution_phase=self.params.resolution_voxel_size_phase,
            img_resolution_slice=self.params.resolution_slice_thickness,
            etl=self.params.etl,
            os_factor=self.params.oversampling,
            read_dir=self.params.read_dir,
            acc_factor_phase=self.params.acceleration_factor,
            acc_read=False,
            te=self.te
        )
        self.recon_params_set = True

    def _set_nav_parameters(self):
        log_module.debug(f"set pypsi recon nav")
        if self.params.use_navs:
            # recon
            self.interface.recon.set_navigator_params(
                lines_per_nav=int(self.params.resolution_n_phase * self.nav_resolution_factor / 2),
                num_of_nav=self.params.number_central_lines + self.params.number_outer_lines,
                nav_acc_factor=2, nav_resolution_scaling=self.nav_resolution_factor,
                num_of_navs_per_tr=2
            )
        else:
            pass

    # emc
    def _set_emc_parameters(self):
        pass
    #     log_module.debug(f"set pypsi emc")
    #     # spawn emc obj with relevant information - to make use of postinit
    #     self.interface.emc = pypsi.parameters.EmcParameters(
    #         gamma_hz=self.interface.specs.gamma,
    #         etl=self.params.etl,
    #         esp=self.params.esp,
    #         bw=self.params.bandwidth,
    #         excitation_angle=self.params.excitation_rf_fa,
    #         excitation_phase=self.params.excitation_rf_phase,
    #         gradient_excitation=self._set_grad_for_emc(self.block_excitation.grad_slice.slice_select_amplitude),
    #         duration_excitation=self.params.excitation_duration,
    #         refocus_angle=self.params.refocusing_rf_fa,
    #         refocus_phase=self.params.refocusing_rf_phase,
    #         gradient_refocus=self._set_grad_for_emc(self.block_refocus.grad_slice.slice_select_amplitude),
    #         duration_refocus=self.params.refocusing_duration,
    #         gradient_crush=self._set_grad_for_emc(self.block_refocus.grad_slice.amplitude[1]),
    #         duration_crush=np.sum(np.diff(self.block_refocus.grad_slice.t_array_s[-4:])) * 1e6,
    #         tes=self.te
    #     )
    #     self._fill_emc_info()

    @abc.abstractmethod
    def _fill_emc_info(self):
        log_module.debug(f"fill pypsi emc")
        # to be implemented for each variant
        pass

    # pulse
    def _set_pulse_info(self):
        # log_module.debug(f"set pypsi pulse")
        # blocks = [self.block_excitation, self.block_refocus]
        # attributes = ["excitation", "refocusing"]
        # for k in range(len(blocks)):
        #     block = blocks[k]
        #     attri = attributes[k]
        #     self.interface.pulse.__setattr__(
        #         attri,
        #         pypsi.parameters.rf_params.RFPulse(
        #             name=attri, bandwidth_in_Hz=block.rf.bandwidth_hz,
        #             duration_in_us=block.rf.t_duration_s * 1e6,
        #             time_bandwidth=block.rf.t_duration_s * block.rf.bandwidth_hz,
        #             num_samples=block.rf.signal.shape[0],
        #             amplitude=np.abs(block.rf.signal),
        #             phase=np.angle(block.rf.signal)
        #         )
        #     )
        pass

    # inits
    def _set_system_specs(self) -> Opts:
        # catch some errors
        if self.specs.rise_time < 1e-6:
            self.specs.rise_time = None

        return Opts(
            B0=self.specs.b_0,
            adc_dead_time=self.specs.adc_dead_time,
            gamma=self.specs.gamma,
            grad_raster_time=self.specs.grad_raster_time,
            grad_unit=self.specs.grad_unit,
            max_grad=self.specs.max_grad,
            max_slew=self.specs.max_slew,
            rf_dead_time=self.specs.rf_dead_time,
            rf_raster_time=self.specs.rf_raster_time,
            rf_ringdown_time=self.specs.rf_ringdown_time,
            rise_time=self.specs.rise_time,
            slew_unit=self.specs.slew_unit
        )

    # methods
    def _set_k_space(self):
        if self.params.acceleration_factor > 1.1:
            # calculate center of k space and indexes for full sampling band
            k_central_phase = round(self.params.resolution_n_phase / 2)
            k_half_central_lines = round(self.params.number_central_lines / 2)
            # set indexes for start and end of full k space center sampling
            k_start = k_central_phase - k_half_central_lines
            k_end = k_central_phase + k_half_central_lines

            # different sampling choices ["weighted_sampling", "interleaved_lines", "grappa"]
            if self.params.sampling_pattern.startswith("weighted"):
                # The rest of the lines we will use tse style phase step blip between the echoes of one echo train
                # Trying random sampling, ie. pick random line numbers for remaining indices,
                # we dont want to pick the same positive as negative phase encodes to account
                # for conjugate symmetry in k-space.
                # Hence, we pick from the positive indexes twice (thinking of the center as 0)
                # without allowing for duplexes and negate half the picks
                # calculate indexes
                k_remaining = np.arange(0, k_start)
                # build array with dim [num_slices, num_outer_lines] to sample different random scheme per slice
                weighting_factor = np.clip(self.params.sample_weighting, 0.01, 1)
                log_module.info(f"\t\t-weighted random sampling of k-space phase encodes, factor: {weighting_factor}")
                # random encodes for different echoes - random choice weighted towards center
                weighting = np.clip(np.power(np.linspace(0, 1, k_start), weighting_factor), 1e-5, 1)
                weighting /= np.sum(weighting)
                for idx_echo in range(self.params.etl):
                    # same encode for all echoes -> central lines
                    self.k_pe_indexes[idx_echo, :self.params.number_central_lines] = np.arange(k_start, k_end)
                    # outer ones sampled from the density distribution weighting
                    k_indices = self.prng.choice(
                        k_remaining,
                        size=self.params.number_outer_lines,
                        replace=False,
                        p=weighting

                    )
                    k_indices[::2] = self.params.resolution_n_phase - 1 - k_indices[::2]
                    self.k_pe_indexes[idx_echo, self.params.number_central_lines:] = np.sort(k_indices)
            elif self.params.sampling_pattern.startswith("interleaved"):
                # we want to skip a line per echo, to achieve complementary lines throughout the echo train
                for idx_echo in range(self.params.etl):
                    # same encode for all echoes -> central lines
                    self.k_pe_indexes[idx_echo, :self.params.number_central_lines] = np.arange(k_start, k_end)
                    # outer ones given by skipping lines
                    # acc factor needs to be integer
                    acc_fact = int(np.round(self.params.acceleration_factor))
                    line_shift = int(idx_echo % acc_fact)
                    k_indices = np.concatenate(
                        (
                            np.arange(line_shift, k_start, acc_fact),
                            np.arange(k_end + line_shift, self.params.resolution_n_phase, acc_fact)
                        )
                    )
                    # broadcast (from rounding errors)
                    len_to_fill = self.k_pe_indexes[idx_echo, self.params.number_central_lines:].shape[0]
                    if k_indices.shape[0] < len_to_fill:
                        if line_shift < 2:
                            # add end line
                            k_indices = np.concatenate((k_indices, np.array([self.params.resolution_n_phase - 1])))
                        else:
                            # add start line
                            k_indices = np.concatenate((k_indices, np.array([0])))
                    elif k_indices.shape[0] > len_to_fill:
                        k_indices = k_indices[:len_to_fill]
                    self.k_pe_indexes[idx_echo, self.params.number_central_lines:] = k_indices
            else:
                log_module.info(f"\t\t-grappa style alternating k-space phase encodes")
                # same encode for all echoes -> central lines
                self.k_pe_indexes[:, :self.params.number_central_lines] = np.arange(k_start, k_end)[None]
                # pick every nth pe
                k_indices = np.arange(0, self.params.resolution_n_phase, int(self.params.acceleration_factor))
                # drop the central ones
                k_indices = k_indices[(k_indices < k_start) | (k_indices > k_end)]
                self.k_pe_indexes[:, self.params.number_central_lines:] = np.sort(k_indices)[None]

        else:
            self.k_pe_indexes[:, :] = np.arange(
                self.params.number_central_lines + self.params.number_outer_lines
            )

    def _set_navigators(self):
        # we use two navigators for now, at the end of each slice slab,
        # could in principle make this gap dependent and acquire per gap if big enough to get a 3D nav volume
        self.nav_num: int = 2
        self.nav_resolution = self.params.resolution_voxel_size_read * self.nav_resolution_factor
        self.nav_slice_thickness = self.params.resolution_slice_thickness * self.nav_resolution_factor

        # create blocks
        self.block_nav_excitation: kernels.Kernel = self._set_nav_excitation()
        self.block_list_nav_acq: list = self._set_nav_acquisition()
        self.id_acq_nav = "nav_acq"

        if self.params.visualize:
            self.block_nav_excitation.plot(path=self.interface.config.output_path, name="nav_excitation")

            for k in range(3):
                self.block_list_nav_acq[k].plot(path=self.interface.config.output_path, name=f"nav_acq_{k}")

        # register sampling trajectories
        # need 2 trajectory lines for navigators: plus + minus directions
        # sanity check that pre-phasing for odd and even read lines are same, i.e. cycling correct
        grad_read_exc_pre = np.sum(self.block_nav_excitation.grad_read.area)
        grad_read_2nd_pre = grad_read_exc_pre + np.sum(
            self.block_list_nav_acq[0].grad_read.area
        )
        grad_read_3rd_pre = grad_read_2nd_pre + np.sum(self.block_list_nav_acq[1].grad_read.area)
        grad_read_4th_pre = grad_read_3rd_pre + np.sum(
            self.block_list_nav_acq[2].grad_read.area
        )
        if np.abs(grad_read_exc_pre - grad_read_3rd_pre) > 1e-9:
            err = f"navigator readout prephasing gradients of odd echoes do not coincide"
            log_module.error(err)
            raise ValueError(err)
        if np.abs(grad_read_2nd_pre - grad_read_4th_pre) > 1e-9:
            err = f"navigator readout prephasing gradients of even echoes do not coincide"
            log_module.error(err)
            raise ValueError(err)
        # register trajectories
        # odd
        acq_nav_block = self.block_list_nav_acq[0]
        self._register_k_trajectory(
            acq_nav_block.get_k_space_trajectory(
                pre_read_area=grad_read_exc_pre,
                fs_grad_area=int(self.params.resolution_n_read / self.nav_resolution_factor) * self.params.delta_k_read
            ),
            identifier=f"{self.id_acq_nav}_odd"
        )
        # even
        acq_nav_block = self.block_list_nav_acq[1]
        self._register_k_trajectory(
            acq_nav_block.get_k_space_trajectory(
                pre_read_area=grad_read_2nd_pre,
                fs_grad_area=int(self.params.resolution_n_read / self.nav_resolution_factor) * self.params.delta_k_read
            ),
            identifier=f"{self.id_acq_nav}_even"
        )

        # calculate timing
        # time for fid navs - one delay in between
        self.nav_t_total = np.sum(
            [b.get_duration() for b in self.block_list_nav_acq]
        ) + np.sum(
            [b.get_duration() for b in self.block_list_nav_acq[:-1]]
        )
        log_module.info(f"\t\t-total fid-nav time (2 navs + 1 delay of 10ms): {self.nav_t_total * 1e3:.2f} ms")

    def _set_nav_excitation(self) -> kernels.Kernel:
        # use excitation kernel without spoiling - only rephasing
        k_ex = kernels.Kernel.excitation_slice_sel(
            pyp_interface=self.params,
            system=self.system,
            use_slice_spoiling=False
        )
        # set up prephasing gradient for fid readouts
        # get timings
        t_spoiling = np.sum(np.diff(k_ex.grad_slice.t_array_s[-4:]))
        t_spoiling_start = k_ex.grad_slice.t_array_s[-4]
        # get area - delta k stays equal since FOV doesnt change
        num_samples_per_read = int(self.params.resolution_n_read / self.nav_resolution_factor)

        grad_read_area = events.GRAD.make_trapezoid(
            channel=self.params.read_dir, system=self.system,
            flat_area=num_samples_per_read * self.params.delta_k_read,
            flat_time=self.params.dwell * num_samples_per_read * self.params.oversampling
        ).area
        # need half of this area (includes ramps etc) to preaphse (negative)
        grad_read_pre = events.GRAD.make_trapezoid(
            channel=self.params.read_dir, system=self.system, area=-grad_read_area / 2,
            duration_s=float(t_spoiling), delay_s=t_spoiling_start
        )
        k_ex.grad_read = grad_read_pre
        return k_ex

    def _set_nav_acquisition(self) -> list:
        # want to use an EPI style readout with acceleration. i.e. skipping of every other line.
        acceleration_factor = 2
        # want to go center out. i.e:
        # acquire line [0, 1, -2, 3, -4, 5 ...] etc i.e. acc_factor_th of the lines + 1,
        pe_increments = np.arange(
            1, int(self.params.resolution_n_phase / self.nav_resolution_factor), acceleration_factor
        )
        pe_increments *= np.power(-1, np.arange(pe_increments.shape[0]))
        # in general only nth of resolution
        block_fid_nav = [kernels.Kernel.acquisition_fid_nav(
            params=self.params,
            system=self.system,
            line_num=k,
            reso_degrading=1 / self.nav_resolution_factor
        ) for k in range(int(self.params.resolution_n_phase / self.nav_resolution_factor / 2))]
        # add spoiling
        block_fid_nav.append(self.block_spoil_end)
        # add delay
        block_fid_nav.append(kernels.Kernel(system=self.system, delay=events.DELAY.make_delay(delay_s=10e-3)))
        return block_fid_nav

    def _nav_apply_slice_offset(self, idx_nav: int):
        sbb = self.block_nav_excitation
        # find the amplitude at time of RF pulse (assumes constant slice select amplitude)
        grad_slice_amplitude_hz = sbb.grad_slice.amplitude[sbb.grad_slice.t_array_s >= sbb.rf.t_delay_s][0]
        # want to set the navs outside of the slice profile with equal distance to the rest of slices
        if idx_nav == 0:
            # first nav below slice slab
            z = np.min(self.z) - np.abs(np.diff(self.z)[0])
        elif idx_nav == 1:
            # second nav above slice slab
            z = np.max(self.z) + np.abs(np.diff(self.z)[0])
        else:
            err = f"sequence setup for only 2 navigators outside slice slab, " \
                  f"index {idx_nav} was given (should be 0 or 1)"
            log_module.error(err)
            raise ValueError(err)
        sbb.rf.freq_offset_hz = grad_slice_amplitude_hz * z
        # we are setting the phase of a pulse here into its phase offset var.
        # To merge both: given phase parameter and any complex signal array data
        sbb.rf.phase_offset_rad = sbb.rf.phase_rad - 2 * np.pi * sbb.rf.freq_offset_hz * sbb.rf.t_mid

    def _loop_navs(self):
        # loop through all navigators
        for nav_idx in range(self.nav_num):
            self._nav_apply_slice_offset(idx_nav=nav_idx)
            # excitation
            # add block
            self.sequence.add_block(*self.block_nav_excitation.list_events_to_ns())
            # epi style nav read
            # we set up a counter to track the phase encode line, k-space center is half of num lines
            line_counter = 0
            central_line = int(self.params.resolution_n_phase / self.nav_resolution_factor / 2) - 1
            # we set up the phase encode increments
            pe_increments = np.arange(1, int(self.params.resolution_n_phase / self.nav_resolution_factor), 2)
            pe_increments *= np.power(-1, np.arange(pe_increments.shape[0]))
            # we loop through all fid nav blocks (whole readout)
            for b_idx in range(self.block_list_nav_acq.__len__()):
                # get the block
                b = self.block_list_nav_acq[b_idx]
                # if at the end we add a delay
                if (nav_idx == 1) & (b_idx == self.block_list_nav_acq.__len__() - 1):
                    self.sequence.add_block(self.delay_slice.to_simple_ns())
                # otherwise we add the block
                else:
                    self.sequence.add_block(*b.list_events_to_ns())
                # if we have a readout we write to sampling pattern file
                # for navigators we want the 0th to have identifier 0, all minus directions have 1, all plus have 2
                if b_idx % 2:
                    nav_ident = "odd"
                else:
                    nav_ident = "even"
                if b.adc.get_duration() > 0:
                    # track which line we are writing from the incremental steps
                    nav_line_pe = np.sum(pe_increments[:line_counter]) + central_line
                    _ = self._write_sampling_pattern_entry(
                        slice_num=nav_idx, pe_num=nav_line_pe, echo_num=0,
                        acq_type=f"{self.id_acq_nav}_{nav_ident}",
                        echo_type="gre-fid", nav_acq=True
                    )
                    line_counter += 1

    def _set_slice_delay(self, t_total_etl: float):
        """
        want to return the slice delay calculated from the effective TR, effective t_etl and number of slices.
        if we use navigators, the effective TR is the TR diminished by the time navigators take,
        and an additional delay is inserted after the navigator block. the delay between navs is fixed.
        """
        # deminish TR by nav - blocks
        tr_eff = self.params.tr * 1e-3 - self.nav_t_total
        max_num_slices = int(np.floor(tr_eff / t_total_etl))
        log_module.info(f"\t\t-total echo train length: {t_total_etl * 1e3:.2f} ms")
        log_module.info(f"\t\t-desired number of slices: {self.params.resolution_slice_num}")
        log_module.info(f"\t\t-possible number of slices within TR: {max_num_slices}")
        if self.params.resolution_slice_num > max_num_slices:
            msg = f"increase TR or Concatenation needed"
            log_module.error(msg)
            raise ValueError(msg)
        num_delays = self.params.resolution_slice_num
        if self.params.use_navs:
            # we want to add a delay additionally after nav block
            num_delays += 1

        self.delay_slice = events.DELAY.make_delay(
            (tr_eff - self.params.resolution_slice_num * t_total_etl) / num_delays,
            system=self.system
        )
        log_module.info(f"\t\t-time between slices: {self.delay_slice.get_duration() * 1e3:.2f} ms")
        if not self.delay_slice.check_on_block_raster():
            self.delay_slice.set_on_block_raster()
            log_module.info(f"\t\t-adjusting TR delay to raster time: {self.delay_slice.get_duration() * 1e3:.2f} ms")

    def _apply_slice_offset(self, idx_slice: int):
        # set phase and freq offset for all slice select pulse gradient kernels
        for sbb in self.kernel_pulses_slice_select:
            grad_slice_amplitude_hz = sbb.grad_slice.amplitude[sbb.grad_slice.t_array_s >= sbb.rf.t_delay_s][0]
            sbb.rf.freq_offset_hz = grad_slice_amplitude_hz * self.z[idx_slice]
            # we are setting the phase of a pulse here into its phase offset var.
            # To merge both: given phase parameter and any complex signal array data
            sbb.rf.phase_offset_rad = sbb.rf.phase_rad - 2 * np.pi * sbb.rf.freq_offset_hz * sbb.rf.t_mid

    def _set_grad_for_emc(self, grad):
        return 1e3 / self.interface.specs.gamma * grad

    def _calculate_scan_time(self):
        t_total = (self.params.tr) * 1e-3 * (
                self.params.number_central_lines + self.params.number_outer_lines + 1
        )
        log_module.info(f"\t\t-total scan time: {t_total / 60:.1f} min ({t_total:.1f} s)")

    def _set_delta_slices(self):
        # multi-slice
        num_slices = self.params.resolution_slice_num
        # cast from mm
        delta_z = self.params.z_extend * 1e-3
        if self.params.interleaved_slice_acquisition:
            log_module.info("\t\t-set interleaved acquisition")
            # want to go through the slices alternating from beginning and middle
            self.z.flat[:num_slices] = np.linspace((-delta_z / 2), (delta_z / 2), num_slices)
            # reshuffle slices mid+1, 1, mid+2, 2, ...
            self.z = self.z.transpose().flatten()[:num_slices]
        else:
            log_module.info("\t\t-set sequential acquisition")
            self.z = np.linspace((-delta_z / 2), (delta_z / 2), num_slices)
        # find reshuffled slice numbers
        for idx_slice_num in range(num_slices):
            z_val = self.z[idx_slice_num]
            z_pos = np.where(np.unique(self.z) == z_val)[0][0]
            self.trueSliceNum[idx_slice_num] = z_pos

    def _set_name_fov(self) -> str:
        fov_r = int(self.params.resolution_fov_read)
        fov_p = int(self.params.resolution_fov_phase / 100 * self.params.resolution_fov_read)
        fov_s = int(self.params.resolution_slice_thickness * self.params.resolution_slice_num)
        return f"fov{fov_r}-{fov_p}-{fov_s}"

    def _set_name_fa(self) -> str:
        return f"fa{int(self.params.refocusing_rf_fa[0])}"

    # def _plot_grad_moments(self, grad_moments: np.ndarray, dt_in_us: int):
    #     ids = ["gx"] * grad_moments.shape[1] + ["gy"] * grad_moments.shape[1] + ["gz"] * grad_moments.shape[1] + \
    #           ["adc"] * grad_moments.shape[1]
    #     ax_time = np.tile(np.arange(grad_moments.shape[1]) * dt_in_us, 4)
    #     df = pd.DataFrame({
    #         "moments": grad_moments.flatten(), "id": ids,
    #         "time": ax_time
    #     })
    #     plotting.plot_grad_moments(mom_df=df, out_path=self.interface.config.output_path, name="sim_moments")


def setup_sequence_cli(name: str):
    # setup logging
    setup_program_logging(name=f"Pulseq sequence build - {name}")
    # setup parser
    parser, args = setup_parser(
        prog_name=f"pulseq_sequence_{name}",
        dict_config_dataclasses={
            "config": PulseqConfig,
            "parameters": PulseqParameters,
            "system": PulseqSystemSpecs
        })
    # get cli args - config files
    config = PulseqConfig.from_cli(args=args.config, parser=parser)
    config.display()
    # build system specifications

    specs_file = plib.Path(config.system_specs_file).absolute()
    if not specs_file.is_file():
        err = f"Provide valid specifications file. {specs_file} not found"
        log_module.error(err)
        raise FileNotFoundError(err)
    log_module.info(f"load specs file: {specs_file}")
    specs = PulseqSystemSpecs.load(specs_file)
    specs.display()

    params = args.parameters
    # check if pulse file was given in config

    params_file = plib.Path(config.parameter_file).absolute()
    msg = f"Parameters configuration file given as CLI argument or in config file."
    if params_file.is_file():
        msg = f"{msg} Loading: {params_file}"
        log_module.info(msg)
        params = PulseqParameters.load(params_file)
    else:
        msg = f"No {msg} Using default values or explicit CLI input.\nCheck if this is the setup you wanted!"
        log_module.warning(msg)

    rf_file = plib.Path(config.pulse_file).absolute()
    if rf_file.is_file():
        log_module.info(f"Setting pulse file: {rf_file.as_posix()}. Might override setting given in Parameter file.")
        params.rf_file = rf_file
    params.display()

    return parser, config, specs, params

