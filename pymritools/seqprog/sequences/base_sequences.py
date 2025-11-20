import json
import logging
import abc
import pathlib as plib
import pickle
from enum import Enum, auto

from scipy.constants import physical_constants
import numpy as np
import polars as pl
import plotly.graph_objects as go
import plotly.subplots as psub
import plotly.colors as plc
import tqdm

from pymritools.seqprog.core import Kernel, GRAD, ADC, DELAY
from pymritools.config.seqprog import PulseqConfig, PulseqSystemSpecs, PulseqParameters2D, Sampling, PlotSeq
from pymritools.config import setup_program_logging, setup_parser
from pypulseq import Opts, Sequence

from pymritools.seqprog.core.utils import check_raster

log_module = logging.getLogger(__name__)


class SequenceType(Enum):
    SEQUENCE = auto()
    CALIBRATION = auto
    RELOAD = auto()


class AdaptiveSampler:
    def __init__(self, p: np.ndarray):
        """
        Parameters:
        - p: 2D importance distribution
        - n1, n2: dimensions of the sampling space
        """
        self.p = p / p.sum()  # Normalize distribution
        self.n1, self.n2 = p.shape

    @classmethod
    def seq_optimized_ac_sampling_from_dim(cls, n: int, n_ac: int, e: int):
        """
        Create an adative sampler given only the dimensions of the line dimension (without AC lines).
        And number of echoes.

        For this we use an estimated sampling density shape obtained from in-vivo measurements.
        """
        x = np.arange(-n // 2, n // 2)
        # this is the rough density shape we found in the optimization
        # (except for some fft shift to make it peak in the middle)
        y = (x / (n // 2)) ** 2 + 2
        y = np.fft.fftshift(y)
        # set central ac lines 0
        y[(-n_ac // 2 <= x) & (x < n_ac // 2)] = 0
        # to be used as a density for numpy choice
        y = y / np.sum(y)
        # expand the echoes
        p = np.repeat(y[:, None], e, axis=1)

        # return sampler
        return cls(p=p)

    def sample(self, n_sub: int, repulsion_radius: int = 2):
        """
        Sample with importance and spatial diversity

        Args:
        - n_sub: number of samples to draw per dimension
        - repulsion_radius: Minimum distance between selected points
        """
        repulsion_radius = max(repulsion_radius, 2)
        # Initial sampling based on importance
        samples = []

        for e in range(self.n2):
            # Candidate points for this dimension
            e_candidates = np.where(self.p[:, e] > 0)[0]

            # Weighted sampling of candidates
            e_weights = self.p[e_candidates, e]
            e_weights /= e_weights.sum()

            # Track selected points for this dimension
            dim_samples = []

            while len(dim_samples) < n_sub:
                # Sample candidate with probability proportional to importance
                candidate_idx = np.random.choice(e_candidates.shape[0], p=e_weights)
                candidate = e_candidates[candidate_idx]

                # Check repulsion condition
                if not self._is_too_close(candidate, dim_samples, repulsion_radius):
                    dim_samples.append(candidate)

            samples.append(dim_samples)

        return np.array(samples)

    @staticmethod
    def _is_too_close(point, existing_points, radius):
        """
        Check if point is too close to existing points
        """
        return any(abs(point - ex) < radius for ex in existing_points)


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

    def __init__(self, config: PulseqConfig, specs: PulseqSystemSpecs, params: PulseqParameters2D,
                 create_excitation_kernel: bool = True, create_refocus_1_kernel: bool = True,
                 create_refocus_kernel: bool = True, relax_read_grad_stress: bool = False, mese_sequence: bool = True):

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
        self.params: PulseqParameters2D = params
        log_module.info(
            f"\n\t\tBandwidth: {self.params.bandwidth:.3f} Hz/px;\n"
            f"\t\tReadout time: {self.params.acquisition_time * 1e3:.1f} ms;\n"
            f"\t\tDwellTime: {self.params.readout_dwell_us:.1f} us;\n"
            f"\t\tNumber of Freq Encodes: {self.params.resolution_n_read};¸\n"
            f"\t\tOversampling: {self.params.oversampling}"
        )
        # make refocusing pulses list of length etl
        # get system specs
        self.specs: PulseqSystemSpecs = specs
        # ToDo: introduce a relax factor to relax gradient stress and pns in parameters
        # easy implementation is to just lower the max slew rate, this way we affect all gradients
        self.system: Opts = self._set_system_specs()

        # for mese sequences we get into trouble if any delays set between excitation and
        # first refocus or between refocus and adc are not on the grad raster ->
        # automatically will be prolonged and lead to sequence timing errors.
        # easy way to prevent this is to just manually double the grad raster in the system and
        # make sure the RF lengths are divisible by this new value,
        # since we dont want this to be shipped to the global pypulseq object we set this first
        # set pypulseq sequence as var -> thats actually whats been build throughout the code and
        # later shipped to a .seq file
        self.sequence: Sequence = Sequence(system=self.system)
        self.sequence_parsed: Sequence = Sequence(system=self.system)
        # we add a second sequence object, we can use this to do some slice profile checks
        self.sequence_calibration = Sequence(system=self.system)
        self.use_calibration_seq: bool = False

        if mese_sequence:
            log_module.info(f"\t\tmulti-echo sequence flag set - ensuring grad raster delays!")
            adopt_raster_time = 2  * self.system.grad_raster_time
            exc_dur = self.params.excitation_duration * 1e-6
            if not check_raster(exc_dur, adopt_raster_time):
                dur = int(np.ceil(exc_dur / adopt_raster_time)) * adopt_raster_time
                self.params.excitation_duration = int(1e6 * dur)
            ref_dur = self.params.refocusing_duration * 1e-6
            if not check_raster(ref_dur, adopt_raster_time):
                dur = int(np.ceil(ref_dur / adopt_raster_time)) * adopt_raster_time
                self.params.refocusing_duration = int(1e6 * dur)
        self.block_list_refocusing = []
        # set sampling object as var -> allows to register k-space trajectories and sampling pattern
        self.sampling: Sampling = Sampling()

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
        # set basic kernels
        # acquisition
        self.block_acquisition: Kernel = Kernel.acquisition_fs(
            params=self.params,
            system=self.system,
            relax_grad_stress=relax_read_grad_stress
        )
        # refocusing - after first
        if create_refocus_kernel:
            self.block_refocus, self.t_spoiling_pe = Kernel.refocus_slice_sel_spoil(
                params=self.params,
                system=self.system,
                pulse_num=1,
                return_pe_time=True,
                read_gradient_to_prephase=self.block_acquisition.grad_read.area / 2,
                pulse_file=self.config.pulse_file_refocusing
            )
        else:
            self.block_refocus: Kernel = NotImplemented
        if create_refocus_1_kernel:
            # refocusing first
            self.block_refocus_1 = Kernel.refocus_slice_sel_spoil(
                params=self.params,
                system=self.system,
                pulse_num=0,
                return_pe_time=False,
                read_gradient_to_prephase=self.block_acquisition.grad_read.area / 2,
                pulse_file=self.config.pulse_file_refocusing
            )
        else:
            self.block_refocus_1: Kernel = NotImplemented
        if create_excitation_kernel:
            # excitation
            # via kernels we can build slice selective blocks of excitation and refocusing
            # if we leave the spoiling gradient of the first refocus (above) we can merge this into the excitation
            # gradient slice refocus gradient. For this we need to now the ramp area of the
            # slice selective refocus 1 gradient in order to account for it. S.th. the slice refocus gradient is
            # equal to the other refocus spoiling gradients used and is composed of: spoiling, refocusing and
            # accounting for ramp area
            ramp_area = float(self.block_refocus_1.grad_slice.area[0])
            spoiling_moment = -self.block_refocus.grad_slice.area[-1]
            # excitation pulse
            self.block_excitation = Kernel.excitation_slice_sel(
                params=self.params, system=self.system,
                adjust_ramp_area=ramp_area,
                spoiling_moment=spoiling_moment,
                pulse_file=self.config.pulse_file_excitation
            )
        else:
            self.block_excitation: Kernel = NotImplemented

        # set up spoling at end of echo train
        self.block_spoil_end: Kernel = Kernel.spoil_all_grads(
            params=self.params, system=self.system
        )

        # save kernels for emc
        self.kernels_to_save: dict = {}

        # use random state for reproducibility of eg sampling patterns
        # (for that matter any random or semi random sampling used)
        self.prng = np.random.RandomState(0)

        # to check interface set
        self.sampling_pattern_set: bool = False
        self.k_trajectory_set: bool = False

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
        # self.navs_on: bool = self.params.use_navs
        self.navs_on: bool = False  # not yet implemented
        self.nav_num: int = 0
        self.nav_t_total: float = 0.0
        # for now we fix the navigator resolution at 5 times coarser than the chosen resolution
        # of scan read direction. Will be different from scan if not isotropic in plane is used
        self.nav_resolution_factor: int = 5
        if self.navs_on:
            self._set_navigators()

    def set_on_grad_raster_time(self, time: float):
        return np.ceil(time / self.system.grad_raster_time) * self.system.grad_raster_time

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
    def write(self, name: str = ""):
        """
        Function to drop out all necessary files:
        .seq file
        pulseq_config.json -> basically the parameter configurations used to produce the sequence
        sampling_config.pkl -> object to store sampling trajectories and echo information

        if no pulse file was given we write one
        ToDo: Port to .h5 format to put this in a common storage and dropout one file
        """
        path = plib.Path(self.config.out_path).absolute()
        if not name:
            name = f"seq"
        voxel_size = ""
        for k in range(3):
            voxel_size += f"-{self.params.get_voxel_size()[k]:.2f}"
        name = (f"{name}_v{self.config.version}_"
                f"acc-{self.params.acceleration_factor:.1f}_"
                f"res{voxel_size}").replace(".", "p")
        if self.params.rf_adapt_z:
            name = f"{name}_rf-ad-z"
        # write sequence file
        save_file = path.joinpath(f"{name}_sequence").with_suffix(".seq")
        log_module.info(f"writing file: {save_file.as_posix()}")
        # write sequence after setting some header definitions (i.e. for correct FOV display on scanner
        self.set_pulseq_definitions()
        _, err_report = self.sequence.write(save_file.as_posix())
        log_module.info(f"reading parsed file: {save_file.as_posix()}")
        self.sequence_parsed.read(file_path=save_file.as_posix(), detect_rf_use=True)

        # write calibration sequence if used
        if self.use_calibration_seq:
            # write sequence file
            save_file = path.joinpath(f"{name}_calibration_sequence").with_suffix(".seq")
            log_module.info(f"writing file: {save_file.as_posix()}")
            # write sequence after setting some header definitions (i.e. for correct FOV display on scanner
            self.set_pulseq_definitions(calib=True)
            self.sequence_calibration.write(save_file.as_posix())


        # write error report
        if err_report is not None:
            d = [err.__dict__ for err in err_report]
            save_file = path.joinpath(f"{name}_pulseq_report").with_suffix(".json")
            log_module.info(f"writing file: {save_file.as_posix()}")
            with open(save_file.as_posix(), "w") as f:
                json.dump(d, f)

        # write pulseq file
        save_file = path.joinpath(f"{name}_pulseq_config").with_suffix(".json")
        log_module.info(f"writing file: {save_file.as_posix()}")
        self.params.save_json(save_file.as_posix(), indent=2)

        # write sampling file
        save_file = path.joinpath(f"{name}_sampling_config").with_suffix(".pkl")
        # log_module.info(f"writing file: {save_file.as_posix()}")
        self.sampling.save(save_file.as_posix())

        # write used kernels
        save_file = path.joinpath(f"{name}_kernels").with_suffix(".pkl")
        log_module.info(f"writing file: {save_file.as_posix()}")
        with open(save_file.as_posix(), "wb") as f:
            pickle.dump(self.kernels_to_save, f)

        # tes
        save_file = path.joinpath(f"{name}_te").with_suffix(".json")
        log_module.info(f"writing file: {save_file.as_posix()}")
        with open(save_file.as_posix(), "w") as f:
            json.dump(self.te, f, indent=2)

        # ToDo: write pulse file
        # if not self.config.pulse_file:
        #     save_file = file_name.joinpath(f"{name}_pulse_shape").with_suffix(".pkl")
        #     log_module.info(f"writing file: {save_file.as_posix()}")
        #     self.save(save_file.as_posix())
        # ToDo: include rf-scaling-z

        # self._check_interface_set()

        # def write_pypsi(self, name: str = ""):
        #     path = plib.Path(self.interface.config.output_path).absolute()
        #     if not name:
        #         name = f"{self.params.name}_{self.params.version}"
        #     name = f"pypsi_{name}"
        #     save_file = path.joinpath(name).with_suffix(".pkl")
        #     log_module.info(f"writing file: {save_file.as_posix()}")
        #     self._check_interface_set()
        #     # write
        #     self.interface.save(save_file.as_posix().__str__())
        #
        #     name = f"z-adapt-rf_{name}"
        #     save_file = path.joinpath(name).with_suffix(".json")
        #     log_module.info(f"writing file: {save_file.as_posix()}")
        #     j_dict = {
        #         "rf_scaling_z": self.rf_slice_adaptive_scaling.tolist(),
        #         "z_slice_idx": np.arange(self.params.resolution_slice_num).tolist()}
        #     with open(save_file.as_posix(), "w") as j_file:
        #         json.dump(j_dict, j_file, indent=2)

    def set_pulseq_definitions(self, calib: bool = False):
        seq = self.sequence if not calib else self.sequence_calibration
        seq.set_definition(
            "FOV",
            [*self.params.get_fov()]
        )
        seq.set_definition(
            "Name",
            f"mese_{self.config.version}".replace(".", "p")
        )
        seq.set_definition(
            "AdcRasterTime",
            1e-07
        )
        seq.set_definition(
            "GradientRasterTime",
            self.specs.grad_raster_time
        )
        seq.set_definition(
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
        if self.config.visualize:
            self.plot_grad_moments(grad_moments, dt_in_us=dt_steps_us)

    def build(self):
        log_module.info(f"__Build Sequence__")
        log_module.info(f"build -- calculate total scan time")
        self._calculate_scan_time()
        log_module.info(f"build -- set up k-space sampling")
        self._set_k_space_sampling()
        log_module.info(f"build -- set up slices")
        self._set_delta_slices()
        log_module.info(f"build variant specifics")
        self._build_variant()
        log_module.info(f"build -- loop lines")
        # prescan for noise correlation
        self._noise_pre_scan()
        self._loop_lines()
        log_module.info(f"build -- profiler sequence")
        self._loop_calibration_sequence()
        log_module.info(f"set recon info data")
        # sampling + k traj
        self._set_k_trajectories()  # raises error if not implemented
        # self._write_sampling_pattern()
        # recon
        # self._set_recon_parameters_img()
        # self._set_nav_parameters()  # raises error if not implemented
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
    def _loop_slices(self, idx_pe_n: int, no_adc: bool = False):
        # to be implemented for each variant, looping through the phase encodes
        pass

    @abc.abstractmethod
    def _loop_calibration_sequence(self):
        # to be implemented for each variant, looping through the additional sequence object for calibration
        pass

    def _noise_pre_scan(self):
        # make delay
        post_delay = DELAY.make_delay(delay_s=0.05, system=self.system)
        # build adc block
        acq = ADC.make_adc(
            system=self.system, num_samples=1000, dwell=self.params.dwell, delay_s=self.system.adc_dead_time
        )
        # use number of noise scans
        for k in range(self.params.number_noise_scans):
            # add to sequence
            self.sequence.add_block(acq.to_simple_ns())
            # write as sampling entry
            self._write_sampling_pattern_entry(
                slice_num=0, pe_num=0, echo_num=k, echo_type="noise_scan", acq_type="noise_scan"
            )
            self.sequence.add_block(post_delay.to_simple_ns())

    def _set_fa_and_update_slice_offset(
            self, rf_idx: int, slice_idx: int, excitation: bool = False, no_ref_1: bool = False
    ):
        if excitation:
            sbb = self.block_excitation
            fa_rad = self.params.excitation_rf_rad_fa
            phase_rad = self.params.excitation_rf_rad_phase
        else:
            # if rf_idx == 0 and not no_ref_1:
            #     sbb = self.block_refocus_1
            # else:
            #     sbb = self.block_refocus
            sbb = self.block_list_refocusing[rf_idx]
            # take flip angle in radiant from options
            fa_rad = self.params.refocusing_rf_rad_fa[rf_idx]
            # take phase as given in options
            phase_rad = self.params.refocusing_rf_rad_phase[rf_idx]
        # calculate the flip angle provided by rf shape
        # flip = sbb.rf.t_duration_s / sbb.rf.signal.shape[0] * np.sum(np.abs(sbb.rf.signal)) * 2 * np.pi
        flip = sbb.rf.t_duration_s / sbb.rf.signal.shape[0] * np.sum(sbb.rf.signal) * 2 * np.pi
        # set rf to produce actually wanted fa.
        # additionally we can multiply with slice adaptive fa scaling - we need true slice position here!
        sbb.rf.signal *= fa_rad / flip * self.rf_slice_adaptive_scaling[self.trueSliceNum[slice_idx]]
        # set phase (the slice dependent phase is set in another method)
        sbb.rf.phase_rad = phase_rad
        self._apply_slice_offset(sbb=sbb, idx_slice=slice_idx)

    def _apply_slice_offset(self, sbb: Kernel, idx_slice: int):
        grad_slice_amplitude_hz = -sbb.grad_slice.amplitude[sbb.grad_slice.t_array_s >= sbb.rf.t_delay_s][0]
        sbb.rf.freq_offset_hz = grad_slice_amplitude_hz * self.z[idx_slice]
        # we are setting the phase of a pulse here into its phase offset var.
        # To merge both: given phase parameter and any complex signal array data
        phase_shift_time = sbb.rf.t_mid
        phase_shift = 2 * np.pi * sbb.rf.freq_offset_hz * phase_shift_time
        sbb.rf.phase_offset_rad = sbb.rf.phase_rad - phase_shift

    def _set_phase_grad(self, echo_idx: int, phase_idx: int,
                        no_ref_1: bool = False, excitation: bool = False,
                        add_rephase: float = 0.0):
        # caution we assume trapezoidal phase encode gradients
        area_factors = np.array([0.5, 1.0, 0.5])
        # we get the actual line index from the sampling pattern, dependent on echo number and phase index in the loop
        idx_phase = self.k_pe_indexes[echo_idx, phase_idx]
        # additionally we need the last blocks phase encode for rephasing
        if echo_idx == 0 and not no_ref_1:
            block = self.block_list_refocusing[0]
        elif echo_idx == 0 and excitation:
            block = self.block_excitation
        else:
            # if we are not on the first refocusing or the first refocusing is identical to the others:
            # we need the last phase encode value to reset before refocusing
            if echo_idx == 0:
                # in case there is no difference in refocusing, the first refocus needs to rephase the excitation phase
                last_idx_phase = self.k_pe_indexes[echo_idx, phase_idx]
            else:
                # we need to rephase the previous phase encode
                last_idx_phase = self.k_pe_indexes[echo_idx - 1, phase_idx]
            block = self.block_list_refocusing[echo_idx]
            if block.grad_phase.amplitude.shape[0] not in [4, 8]:
                warning = (f"Assumes trapezoidal phase encode gradients with ramp up, flat and ramp down, "
                           f"but found different gradient shape ({block.grad_phase.amplitude.shape[0]})!"
                           f" Check Phase encode Gradient shape!")
                log_module.warning(warning)
            # we set the re-phase phase encode gradient - add additional area if set
            phase_enc_time_pre_pulse = np.sum(np.diff(block.grad_phase.t_array_s[:4]) * area_factors)
            block.grad_phase.amplitude[1:3] = (self.phase_areas[
                                                   last_idx_phase] + add_rephase) / phase_enc_time_pre_pulse

        # we set the post pulse phase encode gradient that sets up the next readout
        if np.abs(self.phase_areas[idx_phase]) > 1:
            # we get the time of the phase encode after pulse for every event
            phase_enc_time_post_pulse = np.sum(np.diff(block.grad_phase.t_array_s[-4:]) * area_factors)
            block.grad_phase.amplitude[-3:-1] = - self.phase_areas[idx_phase] / phase_enc_time_post_pulse
        else:
            block.grad_phase.amplitude = np.zeros_like(block.grad_phase.amplitude)

    def _set_end_spoil_phase_grad(self):
        factor = np.array([0.5, 1.0, 0.5])

        # get phase moment of last phase encode
        pe_last_area = np.trapezoid(
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

    # sampling & k - space
    # ToDo: use EPI style phase steps in between GRE readouts to inject different sampling patterns per GRE
    def _write_sampling_pattern_entry(self, slice_num: int, pe_num: int, echo_num: int,
                                      acq_type: str = "", echo_type: str = "", echo_type_num: int = -1,
                                      nav_acq: bool = False):
        log_module.debug(f"set pypsi sampling pattern")
        self.sampling_pattern_set = True
        # register sample in sampling obj
        self.sampling.register_sample(
            scan_num=self.scan_idx, slice_num=slice_num, pe_num=pe_num, echo_num=echo_num,
            acq_type=acq_type, echo_type=echo_type, echo_type_num=echo_type_num,
            nav_acq=nav_acq
        )
        # save to list
        # self._sampling_pattern_constr.append({
        #     "scan_num": self.scan_idx, "slice_num": slice_num, "pe_num": pe_num, "acq_type": acq_type,
        #     "echo_num": echo_num, "echo_type": echo_type, "echo_type_num": echo_type_num,
        #     "nav_acq": nav_acq
        # })
        self.scan_idx += 1
        return echo_type_num + 1

    # def _write_sampling_pattern(self):
    #     self.sampling.sampling_pattern_from_list(sp_list=self._sampling_pattern_constr)

    @abc.abstractmethod
    def _set_k_trajectories(self):
        log_module.debug(f"set k-traj")
        # to be implemented by sequence variants
        pass

    def _register_k_trajectory(self, trajectory: np.ndarray, identifier: str):
        log_module.debug(f"register k - trajectory ({identifier}) in interface")
        self.k_trajectory_set = True
        # build shorthand
        self.sampling.register_trajectory(
            trajectory=trajectory, identifier=identifier
        )

    # emc
    def _set_emc_parameters(self):
        pass

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
    def _set_k_space_sampling(self):
        if self.params.acceleration_factor > 1.1:
            # calculate center of k space and indexes for full sampling band
            k_central_phase = round(self.params.resolution_n_phase / 2)
            k_half_central_lines = round(self.params.number_central_lines / 2)
            # set indexes for start and end of full k space center sampling
            k_start = k_central_phase - k_half_central_lines
            k_end = k_central_phase + k_half_central_lines

            # different sampling choices ["weighted_sampling", "random", "interleaved_lines", "grappa", "optimized]
            if self.params.sampling_pattern.startswith("weighted") or self.params.sampling_pattern.startswith("random"):
                # The rest of the lines we will use tse style phase step blip between the echoes of one echo train
                # Trying random sampling, ie. pick random line numbers for remaining indices,
                # we dont want to pick the same positive as negative phase encodes to account
                # for conjugate symmetry in k-space.
                # Hence, we pick from the positive indexes twice (thinking of the center as 0)
                # without allowing for duplexes and negate half the picks
                # calculate indexes
                k_remaining = np.arange(0, k_start)
                # build array with dim [num_slices, num_outer_lines] to sample different random scheme per slice
                # hardcode weighting
                weighting_factor = 0.5 if self.params.sampling_pattern.startswith("weighted") else 0.0
                log_module.info(
                    f"\t\t-random sampling of k-space phase encodes, central weighting factor: {weighting_factor}")
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
                log_module.info(f"\t\t-interleaved grappa style k-space phase encodes")
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
            elif self.params.sampling_pattern.startswith("optimized"):
                log_module.info(f"\t\t-optimized sampling k-space phase encodes")
                # same encode for all echoes -> central lines
                self.k_pe_indexes[:, :self.params.number_central_lines] = np.repeat(
                    np.arange(k_start, k_end)[None], repeats=self.params.etl, axis=0
                )
                # we want to use an adaptive optimized sampling that does a couple of things:
                # 1) sample a k-space dimension using a density function which concentrates more samples towards
                # the centre (respectively edges of the AC region)
                # 2) try to introduce a radius around each sample in phase encode AND echo dimension
                # to not place adjacent samples (adjacent in the sense of phase and echo directions)
                # not within a certain radius to ensure more uniform sampling across echoes.
                adaptive_sampler = AdaptiveSampler.seq_optimized_ac_sampling_from_dim(
                    n=self.params.resolution_n_phase, n_ac=self.params.number_central_lines,
                    e=self.params.etl
                )

                s_idxs = adaptive_sampler.sample(
                    n_sub=self.params.number_outer_lines, repulsion_radius=int(self.params.acceleration_factor / 2)
                )
                self.k_pe_indexes[:, self.params.number_central_lines:] = np.sort(s_idxs, axis=1)
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
        self.block_nav_excitation: Kernel = self._set_nav_excitation()
        self.block_list_nav_acq: list = self._set_nav_acquisition()
        self.id_acq_nav = "nav_acq"

        if self.config.visualize:
            self.block_nav_excitation.plot(path=self.path_figs, name="nav_excitation")

            for k in range(3):
                self.block_list_nav_acq[k].plot(path=self.path_figs, name=f"nav_acq_{k}")

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

    def _set_nav_excitation(self) -> Kernel:
        # use excitation kernel without spoiling - only rephasing
        k_ex = Kernel.excitation_slice_sel(
            params=self.params,
            system=self.system,
            spoiling_moment=0.0
        )
        # set up prephasing gradient for fid readouts
        # get timings
        t_spoiling = np.sum(np.diff(k_ex.grad_slice.t_array_s[-4:]))
        t_spoiling_start = k_ex.grad_slice.t_array_s[-4]
        # get area - delta k stays equal since FOV doesnt change
        num_samples_per_read = int(self.params.resolution_n_read / self.nav_resolution_factor)

        grad_read_area = GRAD.make_trapezoid(
            channel=self.params.read_dir, system=self.system,
            flat_area=num_samples_per_read * self.params.delta_k_read,
            flat_time=self.params.dwell * num_samples_per_read * self.params.oversampling
        ).area
        # need half of this area (includes ramps etc) to preaphse (negative)
        grad_read_pre = GRAD.make_trapezoid(
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
        block_fid_nav = [Kernel.acquisition_fid_nav(
            params=self.params,
            system=self.system,
            line_num=k,
            reso_degrading=1 / self.nav_resolution_factor
        ) for k in range(int(self.params.resolution_n_phase / self.nav_resolution_factor / 2))]
        # add spoiling
        block_fid_nav.append(self.block_spoil_end)
        # add delay
        block_fid_nav.append(Kernel(system=self.system, delay=DELAY.make_delay(delay_s=10e-3)))
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

        self.delay_slice = DELAY.make_delay(
            (tr_eff - self.params.resolution_slice_num * t_total_etl) / num_delays,
            system=self.system
        )
        log_module.info(f"\t\t-time between slices: {self.delay_slice.get_duration() * 1e3:.2f} ms")
        if not self.delay_slice.check_on_block_raster():
            self.delay_slice.set_on_block_raster()
            log_module.info(f"\t\t-adjusting TR delay to raster time: {self.delay_slice.get_duration() * 1e3:.2f} ms")

    def _set_grad_for_emc(self, grad):
        return 1e3 / self.specs.gamma * grad

    def _calculate_scan_time(self):
        t_total = self.params.tr * 1e-3 * (
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
            # base (sorted) slice positions
            z_lin = np.linspace((-delta_z / 2), (delta_z / 2), num_slices)

            # Build order: [0, mid, 2, mid+2, 4, mid+4, ..., 1, mid+1, 3, mid+3, ...]
            mid = num_slices // 2
            even_first = list(range(0, mid, 2))
            even_second = list(range(mid, num_slices, 2))
            odd_first = list(range(1, mid, 2))
            odd_second = list(range(mid + 1, num_slices, 2))

            order = []
            # interleave even indices from first and second halves
            for i in range(max(len(even_first), len(even_second))):
                if i < len(even_first):
                    order.append(even_first[i])
                if i < len(even_second):
                    order.append(even_second[i])
            # then interleave odd indices from first and second halves
            for i in range(max(len(odd_first), len(odd_second))):
                if i < len(odd_first):
                    order.append(odd_first[i])
                if i < len(odd_second):
                    order.append(odd_second[i])
            # we already know the original indices in `order`, save for later sorting of raw data
            self.trueSliceNum = np.asarray(order, dtype=int)
            # apply order to z positions
            self.z = z_lin[self.trueSliceNum]
        else:
            log_module.info("\t\t-set sequential acquisition")
            self.z = np.linspace((-delta_z / 2), (delta_z / 2), num_slices)
            # sequential: identity mapping
            self.trueSliceNum = np.arange(num_slices)

    def _set_name_fov(self) -> str:
        fov_r = int(self.params.resolution_fov_read)
        fov_p = int(self.params.resolution_fov_phase / 100 * self.params.resolution_fov_read)
        fov_s = int(self.params.resolution_slice_thickness * self.params.resolution_slice_num)
        return f"fov{fov_r}-{fov_p}-{fov_s}"

    def _set_name_fa(self) -> str:
        return f"fa{int(self.params.refocusing_rf_fa[0])}"

    def plot_grad_moments(self, df_grad_moments: pl.DataFrame):
        pass
        # ids = ["gx"] * grad_moments.shape[1] + ["gy"] * grad_moments.shape[1] + ["gz"] * grad_moments.shape[1] + \
        #       ["adc"] * grad_moments.shape[1]
        # ax_time = np.tile(np.arange(grad_moments.shape[1]) * dt_in_us, 4)
        # df = pd.DataFrame({
        #     "moments": grad_moments.flatten(), "id": ids,
        #     "time": ax_time
        # })
        # fig = px.scatter(mom_df, x="time", y="moments", color="id")
        # fig_path = create_fig_dir_ensure_exists(out_path)
        # fig_path = fig_path.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
        # log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
        # if file_suffix in ["png", "pdf"]:
        #     fig.write_image(fig_path.as_posix())
        # else:
        #     fig.write_html(fig_path.as_posix())

    def plot_sequence(self, t_start_s: float = 0.0, t_end_s: float = 10.0,
                      sim_grad_moments: bool = False, file_suffix: str = "html",
                      plot_seq: SequenceType = SequenceType.SEQUENCE
                      ):
        match plot_seq:
            case SequenceType.CALIBRATION:
                seq = self.sequence_calibration
                log_module.info("\t\t-plotting calibration sequence")
                name = "calib_"
            case SequenceType.RELOAD:
                log_module.info("\t\t-plotting sequence from .seq file")
                seq = self.sequence_parsed
                name = "parsed_"
            case _:
                log_module.info("\t\t-plotting sequence")
                seq = self.sequence
                name = ""

        name = f"{name}sequence_t-{int(t_start_s):d}_to_t-{int(t_end_s):d}"
        plot_sequence(
            seq=seq, name=name, path=self.path_figs,
            t_start_s=t_start_s, t_end_s=t_end_s, sim_grad_moments=sim_grad_moments,
            file_suffix=file_suffix, excitation_duration=self.params.excitation_duration
        )

    def plot_sampling(self, file_suffix: str = "html"):
        # trajectories
        df_kt = self.sampling.df_trajectories

        fig = go.Figure()
        for acq in df_kt["acquisition"].unique():
            df_acq = df_kt.filter(pl.col("acquisition") == acq)
            fig.add_trace(
                go.Scatter(
                    x=df_acq["adc_sampling_num"],
                    y=df_acq["k_traj_position"],
                    mode="markers+lines",
                    name=acq
                )
            )
        fig.update_yaxes(title="Sample Position [1/FOV]")
        fig.update_xaxes(title="Sample Number")

        name = f"sequence_trajectories"
        fig_path = self.path_figs.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
        log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
        if file_suffix in ["png", "pdf"]:
            fig.write_image(fig_path.as_posix())
        else:
            fig.write_html(fig_path.as_posix())

        # sampling mask
        len_x = 15
        num_echoes = max([s.echo_number for s in self.sampling.samples if "noise" not in s.acquisition_type]) + 1
        n_pe = self.params.resolution_n_phase
        sampling_mask = np.zeros((len_x * num_echoes, n_pe))

        for i, s in enumerate(self.sampling.samples):
            if "noise" in s.acquisition_type:
                continue
            ne = s.echo_number
            if "bd" in s.acquisition_type:
                m_val = 2
            else:
                m_val = 1
            sampling_mask[ne * len_x:(ne + 1) * len_x, s.phase_encode_number] = m_val
            sampling_mask[(ne + 1) * len_x - 1, :] = 3

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=sampling_mask, transpose=True, zmin=0, zmax=4,
                colorscale="Viridis",
                showscale=False, showlegend=False
            )
        )
        m_vals = np.unique(sampling_mask.astype(int))
        names = ["non sampled", "positive read grad", "negative read grad", ""]
        cmap = plc.sample_colorscale("viridis", 4, 0, 1)
        for i, m in enumerate(m_vals):
            if names[i]:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=names[i],
                        marker=dict(size=7, color=cmap[i], symbol='square'),
                    ),
                )
        tickvals = np.linspace(
            0, sampling_mask.shape[0], num_echoes + 1
        ) + sampling_mask.shape[0] / (num_echoes + 1) / 2
        fig.update_xaxes(title="echo number", tickmode="array", tickvals=tickvals,
                         ticktext=np.arange(1, num_echoes + 1))
        fig.update_yaxes(title="phase encode number")

        name = f"sampling_mask"
        fig_path = self.path_figs.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
        log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
        if file_suffix in ["png", "pdf"]:
            fig.write_image(fig_path.as_posix())
        else:
            fig.write_html(fig_path.as_posix())


# def _plot_grad_moments(self, grad_moments: np.ndarray, dt_in_us: int):
#     ids = ["gx"] * grad_moments.shape[1] + ["gy"] * grad_moments.shape[1] + ["gz"] * grad_moments.shape[1] + \
#           ["adc"] * grad_moments.shape[1]
#     ax_time = np.tile(np.arange(grad_moments.shape[1]) * dt_in_us, 4)
#     df = pd.DataFrame({
#         "moments": grad_moments.flatten(), "id": ids,
#         "time": ax_time
#     })
#     plotting.plot_grad_moments(mom_df=df, out_path=self.interface.config.output_path, name="sim_moments")

def simulate_grad_moments(df_rf_grads: pl.DataFrame):
    log_module.info(f"simulate gradient moments")
    # gradient amplitudes are named after axes
    grad_channels = ['gx', 'gy', 'gz']
    # we have a dataframe with all gradient amplitudes and rf effects on moment [0=None, 1=Null, 2=Flip]
    df_rf_grads = df_rf_grads.sort(by=["time"])
    grad_moments = {
        "GRAD gx": {"moment": [], "time": [], "last_time": 0.0, "last_amp": 0.0},
        "GRAD gy": {"moment": [], "time": [], "last_time": 0.0, "last_amp": 0.0},
        "GRAD gz": {"moment": [], "time": [], "last_time": 0.0, "last_amp": 0.0}
    }
    # we move through the time sorted entries
    for row in tqdm.tqdm(df_rf_grads.iter_rows(), desc="Process gradients"):
        data, time, label = row
        # if we encounter a gradient we calculate the respective moment change
        if label in grad_moments.keys():
            # if no previous moment entry we skip
            if not grad_moments[label]["moment"]:
                # collect the time
                grad_moments[label]["time"].append(time)
                grad_moments[label]["moment"].append(0.0)
            else:
                # cummulate gradient
                grad_moments[label]["moment"].append(
                    grad_moments[label]["moment"][-1] + np.trapz(
                        x=[grad_moments[label]["last_time"], time],
                        y=[grad_moments[label]["last_amp"], data]
                    ) * 1e-6
                )
                grad_moments[label]["time"].append(time)
            grad_moments[label]["last_amp"] = data
            grad_moments[label]["last_time"] = time
        if label == "RF grad moment effect":
            # we calculate the moment until this point for all axes, saved last gradient amp for all gradients,
            # assumes no gradient changes across an RF
            data = int(data)
            for label in grad_moments.keys():
                # if accessed before
                if grad_moments[label]["moment"]:
                    # moment til rf center
                    grad_moments[label]["time"].append(time - 1e-9)
                    grad_moments[label]["moment"].append(
                        grad_moments[label]["moment"][-1] + np.trapezoid(
                            x=[grad_moments[label]["last_time"], time],
                            y=[grad_moments[label]["last_amp"]] * 2
                        ) * 1e-6
                    )
                    # save as last accessed time
                    grad_moments[label]["last_time"] = time
                    # RF effect
                    grad_moments[label]["time"].append(time + 1e-9)
                    if data == 1:
                        # grad mom null
                        grad_moments[label]["moment"].append(0.0)
                    elif data == 2:
                        # grad mom flip
                        grad_moments[label]["moment"].append(-grad_moments[label]["moment"][-1])
                    elif data == 0:
                        # just use last gradient
                        grad_moments[label]["moment"].append(grad_moments[label]["moment"][-1])
                    else:
                        err = f"unable to handle rf gradient moment effect for set data value: {data}"
                        log_module.error(err)
                        raise ValueError(err)
    # build dataframe
    times = []
    moments = []
    labels = []
    for key in grad_moments.keys():
        times.extend(grad_moments[key]["time"])
        moments.extend(grad_moments[key]["moment"])
        labels.extend([key] * len(grad_moments[key]["time"]))
    return pl.DataFrame({
        "time": times, "moment": moments, "labels": labels
    })


def plot_simulate_seq():
    # setup logging
    setup_program_logging(name="Plot .seq", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(
        prog_name="Plot .seq",
        dict_config_dataclasses={"settings": PlotSeq}
    )
    # get config
    settings = PlotSeq.from_cli(args=args.settings, parser=parser)
    settings.display()

    in_file = plib.Path(settings.input_file).absolute()
    if not in_file.is_file():
        err = f"File ({in_file}) does not exist"
        log_module.error(err)
        raise FileNotFoundError(err)

    log_module.info(f"reading parsed file: {in_file.as_posix()}")
    seq = Sequence()
    seq.read(file_path=in_file.as_posix(), detect_rf_use=True)
    plot_sequence(
        seq=seq, name=settings.name, path=settings.path,
        t_start_s=settings.t_start_s, t_end_s=settings.t_end_s,
        sim_grad_moments=settings.sim_grad_moments, file_suffix=settings.file_suffix,
        excitation_duration=settings.excitation_duration
    )


def plot_sequence(seq: Sequence, name: str, path: str,
                  t_start_s: float = 0.0, t_end_s: float = 10.0,
                  sim_grad_moments: bool = False, file_suffix: str = "html",
                  excitation_duration: float = 0.0,
                  ):
    path_figs = plib.Path(path).absolute()
    gamma = physical_constants["proton gyromag. ratio in MHz/T"][0] * 1e6
    logging.debug(f"plot_seq")
    # transform to us
    t_start_us = int(t_start_s * 1e6)
    t_end_us = int(t_end_s * 1e6)
    t_total_us = t_end_us - t_start_us
    if t_total_us < 1:
        err = f"end time needs to be after start time"
        log_module.error(err)
        raise ValueError(err)
    # find starting idx
    start_idx = 0
    t_cum_us = 0

    # go through blocks - find start block
    for block_idx, block_duration in seq.block_durations.items():
        t_cum_us += 1e6 * block_duration
        if t_cum_us > t_start_us:
            start_idx = block_idx
            break
        if block_idx == len(seq.block_durations) - 1:
            err = (f"looped through sequence blocks to get to {t_cum_us} us, "
                   f"and didnt arrive at starting time given {t_start_us} us")
            log_module.error(err)
            raise AttributeError(err)
    t_cum_us = 0
    # set up lists to fill with values
    times = []
    values = []
    labels = []
    # for simulating grad moments, track rf timings
    rf_flip_null = [[], []]

    grad_channels = ['gx', 'gy', 'gz']

    def append_to_lists(time: float | int | list, value: float | list, label: str):
        if isinstance(time, int):
            time = float(time)
            value = float(value)
        if isinstance(time, float):
            times.append(time)
            values.append(value)
            labels.append(label)
        else:
            times.extend(np.array(time).astype(float).tolist())
            values.extend(np.array(value).astype(float).tolist())
            labels.extend([label] * len(time))

    # start with first block after start time
    for block_idx, block_duration in seq.block_durations.items():
        if block_idx < start_idx:
            continue
        if block_idx >= len(seq.block_durations):
            break
        # set start block
        t0 = t_cum_us
        block = seq.get_block(block_idx + 1)
        if t_cum_us + 1e6 * block_duration > t_total_us:
            break
        # add data to the lists
        if getattr(block, 'rf') is not None:
            rf = block.rf
            start = t0 + int(1e6 * rf.delay)
            # starting point at 0
            append_to_lists(start - 1e-6, 0.0, label="RF amp")
            append_to_lists(start - 1e-6, 0.0, label="RF phase")
            t_rf = rf.t * 1e6
            angle = np.angle(
                rf.signal * np.exp(1j * rf.phase_offset) * np.exp(1j * 2 * np.pi * rf.t * rf.freq_offset)
            )
            angle[angle < -np.pi + 1e-7] = np.pi
            signal = np.abs(rf.signal)
            if sim_grad_moments:
                if rf.use == "excitation" or np.abs(rf.shape_dur * 1e6 - excitation_duration) < 1e-3:
                    identifier = 1
                elif rf.use == "refocusing":
                    identifier = 2
                else:
                    err = f"grad moment effect of rf ({rf.use}) not recognized or not implemented."
                    log_module.error(err)
                    raise ValueError(err)
                # assumes rf effect in the center
                append_to_lists(start + rf.shape_dur * 1e6 / 2, identifier,
                                label="RF grad moment effect")

            append_to_lists(time=(t_rf + start).tolist(), value=signal.tolist(), label="RF amp")
            append_to_lists(time=(t_rf + start).tolist(), value=angle.tolist(), label="RF phase")
            # set back to 0
            append_to_lists(t_rf[-1] + start + 1e-6, 0.0, label="RF amp")
            append_to_lists(t_rf[-1] + start + 1e-6, 0.0, label="RF phase")

        for x in range(len(grad_channels)):
            if getattr(block, grad_channels[x]) is not None:
                grad = getattr(block, grad_channels[x])
                if grad.type == 'trap':
                    amp_value = 1e3 * grad.amplitude / gamma
                elif grad.type == 'grad':
                    amp_value = 1e3 * grad.waveform / gamma
                else:
                    amp_value = 0
                start = int(t0 + 1e6 * grad.delay)
                t = grad.tt * 1e6
                append_to_lists(time=start + t, value=amp_value, label=f"GRAD {grad_channels[x]}")

        if getattr(block, 'adc') is not None:
            adc = block.adc
            start = int(t0 + 1e6 * adc.delay)
            # set starting point to 0
            append_to_lists(time=start - 1e-6, value=0.0, label="ADC")
            end = int(start + adc.dwell * adc.num_samples * 1e6)
            dead_time = int(end + 1e6 * adc.dead_time)
            append_to_lists(time=[start, end, end + 1e-6, dead_time], value=[1.0, 1.0, 0.2, 0.2], label="ADC")
            # set end point to 0
            append_to_lists(time=dead_time + 1e-6, value=0.0, label="ADC")
        t_cum_us += int(1e6 * getattr(block, 'block_duration'))

    # build a dataframe storing all data and times
    df = pl.DataFrame({
        "data": values, "time": times, "labels": labels
    })

    # set up figure
    num_rows = 2
    specs = [[{"secondary_y": True}], [{"secondary_y": True}]]

    # simulate grad moments
    if sim_grad_moments:
        # can simulate the moments having a dataframe with corresponding gradient and rf data
        df_grad_moments = simulate_grad_moments(
            df_rf_grads=df.filter(
                pl.col("labels").is_in(
                    [*[f"GRAD {grad_channels[k]}" for k in range(3)], "RF grad moment effect"]
                )
            )
        )
        # self.plot_grad_moments(df_grad_moments=df_grad_moments)
        num_rows += 1
        specs.append([{"secondary_y": False}])
    else:
        df_grad_moments = None

    fig = psub.make_subplots(
        num_rows, 1,
        specs=specs,
        shared_xaxes=True
    )

    # top axis left
    tmp_df = df.filter(pl.col("labels") == "RF amp")
    tmp_df = tmp_df.with_columns(pl.col("data") / pl.col("data").max() * np.pi)

    fig.add_trace(
        go.Scattergl(
            x=tmp_df["time"], y=tmp_df["data"], name="RF Amplitude"
        ),
        row=1, col=1, secondary_y=False
    )

    tmp_df = df.filter(pl.col("labels") == "RF phase")
    fig.add_trace(
        go.Scattergl(
            x=tmp_df["time"], y=tmp_df["data"],
            name="RF Phase [rad]", opacity=0.3
        ),
        row=1, col=1, secondary_y=False
    )

    # axes properties
    fig.update_yaxes(title_text="RF Amplitude & Phase", range=[-3.5, 3.5], row=1, col=1, secondary_y=False)

    # top axis right
    tmp_df = df.filter(pl.col("labels") == "GRAD gz")
    fig.add_trace(
        go.Scattergl(
            x=tmp_df["time"], y=tmp_df["data"], name="Gradient gz"
        ),
        row=1, col=1, secondary_y=True
    )

    fig.update_yaxes(
        title_text="Gradient Slice [mT/m]",
        range=[-1.2 * tmp_df["data"].abs().max(), 1.2 * tmp_df["data"].abs().max()],
        row=1, col=1, secondary_y=True
    )

    # bottom axis left
    tmp_df = df.filter(pl.col("labels") == "ADC")
    fig.add_trace(
        go.Scattergl(
            x=tmp_df["time"], y=tmp_df["data"], name="ADC", fill="tozeroy", opacity=0.5
        ),
        row=2, col=1, secondary_y=False
    )

    fig.update_xaxes(title_text="Time [us]", row=2, col=1)
    fig.update_yaxes(title_text="ADC", range=[-1.5, 1.5], row=2, col=1, secondary_y=False)

    # bottom axis right
    max_val = 40
    for k in range(2):
        tmp_df = df.filter(pl.col("labels") == f"GRAD {grad_channels[k]}")
        fig.add_trace(
            go.Scattergl(
                x=tmp_df["time"], y=tmp_df["data"], name=f"Gradient {grad_channels[k]} [mT/m]"
            ),
            row=2, col=1, secondary_y=True
        )
        if max_val < tmp_df["data"].abs().max():
            max_val = tmp_df["data"].abs().max()
    fig.update_yaxes(
        title_text="Gradient [mT/m]", range=[-1.2 * max_val, 1.2 * max_val],
        row=2, col=1, secondary_y=True
    )

    # add gradient moment simulation
    if sim_grad_moments:
        max_val = df_grad_moments.filter(
            pl.col("labels").is_in(
                [*[f"GRAD {grad_channels[k]}" for k in range(3)]]
            )
        )["moment"].abs().max()

        for k in range(3):
            tmp_df = df_grad_moments.filter(pl.col("labels") == f"GRAD {grad_channels[k]}").sort(by="time")
            fig.add_trace(
                go.Scattergl(
                    x=tmp_df["time"], y=tmp_df["moment"], name=f"Gradient Moment {grad_channels[k]} [mTs/m]"
                ),
                3, 1
            )
        fig.update_yaxes(title_text="Gradient Moment [mT s/m]", range=[-1.2 * max_val, 1.2 * max_val], row=3, col=1,
                         secondary_y=False)
    fig.update_layout(
        width=1000,
        height=800
    )
    fig_path = path_figs.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
    log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
    if file_suffix in ["png", "pdf"]:
        fig.write_image(fig_path.as_posix())
    else:
        fig.write_html(fig_path.as_posix())

    def plot_sampling(self, file_suffix: str = "html"):
        # trajectories
        df_kt = self.sampling.df_trajectories

        fig = go.Figure()
        for acq in df_kt["acquisition"].unique():
            df_acq = df_kt.filter(pl.col("acquisition") == acq)
            fig.add_trace(
                go.Scatter(
                    x=df_acq["adc_sampling_num"],
                    y=df_acq["k_traj_position"],
                    mode="markers+lines",
                    name=acq
                )
            )
        fig.update_yaxes(title="Sample Position [1/FOV]")
        fig.update_xaxes(title="Sample Number")

        name = f"sequence_trajectories"
        fig_path = self.path_figs.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
        log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
        if file_suffix in ["png", "pdf"]:
            fig.write_image(fig_path.as_posix())
        else:
            fig.write_html(fig_path.as_posix())

        # sampling mask
        len_x = 15
        num_echoes = max([s.echo_number for s in self.sampling.samples if "noise" not in s.acquisition_type]) + 1
        n_pe = self.params.resolution_n_phase
        sampling_mask = np.zeros((len_x * num_echoes, n_pe))

        for i, s in enumerate(self.sampling.samples):
            if "noise" in s.acquisition_type:
                continue
            ne = s.echo_number
            if "bd" in s.acquisition_type:
                m_val = 2
            else:
                m_val = 1
            sampling_mask[ne * len_x:(ne + 1) * len_x, s.phase_encode_number] = m_val
            sampling_mask[(ne + 1) * len_x - 1, :] = 3

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=sampling_mask, transpose=True, zmin=0, zmax=4,
                colorscale="Viridis",
                showscale=False, showlegend=False
            )
        )
        m_vals = np.unique(sampling_mask.astype(int))
        names = ["non sampled", "positive read grad", "negative read grad", ""]
        cmap = plc.sample_colorscale("viridis", 4, 0, 1)
        for i, m in enumerate(m_vals):
            if names[i]:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=names[i],
                        marker=dict(size=7, color=cmap[i], symbol='square'),
                    ),
                )
        tickvals = np.linspace(
            0, sampling_mask.shape[0], num_echoes + 1
        ) + sampling_mask.shape[0] / (num_echoes + 1) / 2
        fig.update_xaxes(title="echo number", tickmode="array", tickvals=tickvals,
                         ticktext=np.arange(1, num_echoes + 1))
        fig.update_yaxes(title="phase encode number")

        name = f"sampling_mask"
        fig_path = self.path_figs.joinpath(f"plot_{name}").with_suffix(f".{file_suffix}")
        log_module.info(f"\t\t - writing file: {fig_path.as_posix()}")
        if file_suffix in ["png", "pdf"]:
            fig.write_image(fig_path.as_posix())
        else:
            fig.write_html(fig_path.as_posix())


def setup_sequence_cli(name: str):
    # setup logging
    setup_program_logging(name=f"Pulseq sequence build - {name}")
    # setup parser
    parser, args = setup_parser(
        prog_name=f"pulseq_sequence_{name}",
        dict_config_dataclasses={
            "config": PulseqConfig,
            "parameters": PulseqParameters2D,
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
    msg = f"Parameters configuration file given as CLI argument or in config file."
    if not config.parameter_file:
        msg = f"No {msg}. Using default values or explicit CLI input.\nCheck if this is the setup you wanted!"
        log_module.warning(msg)
    params_file = plib.Path(config.parameter_file).absolute()
    if params_file.is_file():
        log_module.info(msg)
        log_module.info(f"Loading file: {params_file}")
        params = PulseqParameters2D.load(params_file)
    else:
        msg = f"File {params_file} not found"
        log_module.error(msg)
        raise FileNotFoundError(msg)

    # rf_file_exc = plib.Path(config.pulse_file_excitation).absolute()
    #
    # rf_file_ref = plib.Path(config.pulse_file_refocusing).absolute()
    # if rf_file.is_file():
    #     log_module.info(f"Setting pulse file: {rf_file.as_posix()}.")
    #     params.rf_file = rf_file
    params.display()

    return parser, config, specs, params


def build(config: PulseqConfig, sequence: Sequence2D, name: str = ""):
    """
    Function to build the sequence and perform all necessary writing steps
    """
    # build sequence
    sequence.build()
    # get sequence object
    pypulseq_seq_object = sequence.get_pypulseq_seq()
    # sum scan time
    scan_time = np.sum([item[1] for item in pypulseq_seq_object.block_durations.items()])
    logging.info(f"Total Scan Time Sum Seq File: {scan_time / 60:.1f} min")

    logging.info("Verifying and Writing Files")
    path_out = plib.Path(config.out_path).absolute()
    # verifying
    if config.report:
        out_file = path_out.joinpath("report.txt")
        with open(out_file, "w") as w_file:
            report = pypulseq_seq_object.test_report()
            ok, err_rep = pypulseq_seq_object.check_timing()
            log = "report \n" + report + "\ntiming_check \n" + str(ok) + "\ntiming_error \n"
            w_file.write(log)
            for err_rep_item in err_rep:
                w_file.write(f"{str(err_rep_item)}\n")

    # saving .seq file
    sequence.write(name=name)

    if config.visualize:
        logging.info("Plotting")

        sequence.plot_sequence(
            t_start_s=0, t_end_s=50, sim_grad_moments=True, plot_seq=SequenceType.CALIBRATION
        )

        # plot start
        sequence.plot_sequence(
            t_start_s=0, t_end_s=0.3 * sequence.params.tr * 1e-3, sim_grad_moments=True,
            plot_seq=SequenceType.SEQUENCE
        )
        sequence.plot_sequence(
            t_start_s=0, t_end_s=0.3 * sequence.params.tr * 1e-3, sim_grad_moments=True,
            plot_seq=SequenceType.RELOAD
        )
        # if sequence.params.resolution_slice_num < 15:
        #     n = 2
        # elif sequence.params.resolution_slice_num < 30:
        #     n = 1
        # else:
        #     n = 0.5
        # # plot within ac region
        # n_start = int(sequence.params.number_outer_lines // 2) + 4
        # sequence.plot_sequence(
        #     t_start_s=n_start * sequence.params.tr * 1e-3,
        #     t_end_s=(n_start + n) * sequence.params.tr * 1e-3,
        #     sim_grad_moments=True
        # )
        # # plot close to end
        # n_start = int(2 * sequence.params.number_outer_lines // 3) + sequence.params.number_central_lines + 4
        # sequence.plot_sequence(
        #     t_start_s=n_start * sequence.params.tr * 1e-3,
        #     t_end_s=(n_start + n) * sequence.params.tr * 1e-3,
        #     sim_grad_moments=True
        # )
        sequence.plot_sampling()
