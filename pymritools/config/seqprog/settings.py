import logging
from dataclasses import dataclass

from pymritools.config import BaseClass
from simple_parsing import field
from simple_parsing.helpers import Serializable
import numpy as np
log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    system_specs_file: str = field(
        alias="-s", default="",
        help="MRI scanning system specifications file."
    )
    parameter_file: str = field(
        alias="-p", default="",
        help="Parameter configuration for sequence build. Fixing all necessary variables."
    )
    pulse_file_excitation: str = field(
        alias="-rfe", default="",
        help="Provide external pulse shape file to be used for the excitation RF pulses."
    )
    pulse_file_refocusing: str = field(
        alias="-rfr", default="",
        help="Provide external pulse shape file to be used for the refocusing RF pulses."
    )
    # flags
    version: str = field(
        default="1.0",
        help="Set sequence version. Default is 1.0. Usually is done in code releases and does not need to be touched."
    )
    report: bool = field(
        alias="-r", default=False,
        help="Set to True to generate a report of the sequence. Usually slows down .seq generation significantly."
    )


@dataclass
class Parameters2D(Serializable):
    # resolution
    resolution_fov_read: float = field(
        default=212.0, help="FoV in read direction [mm]."
    )
    resolution_fov_phase: float = field(
        default=100.0, help="FoV in phase direction [percentage]."
    )
    resolution_base: int = field(
        default=212, help="Base Resolution, i.e. number of encodes in read direction."
    )
    resolution_slice_thickness: float = field(
        default=1.0, help="Slice thickness of 2D slices [mm]."
    )
    resolution_slice_num: int = field(
        default=30, help="Number of slices."
    )
    resolution_slice_gap: float = field(
        default=200.0, help="Gap between slices, factor of slice thickness [percentage]."
    )

    # PI / acceleration
    number_central_lines: int = field(
        default=36, help="Parallel imaging, number of central lines."
    )
    acceleration_factor: float = field(
        default=4.0, help="Parallel imaging, acceleration factor for outer lines."
    )

    # RF
    rf_adapt_z: bool = field(
        alias="-rfaz", default=False,
        help="Toggle adaptive scaling of RF pulses towards lower slices to counter B1 transmit profile bias effects."
    )

    # RF - excitation
    excitation_rf_fa: float = field(
        default=90.0, help="Excitation pulse flip angle [°]."
    )
    excitation_rf_phase: float = field(
        default=90.0, help="Excitation pulse phase [°]"
    )
    excitation_rf_time_bw_prod: float = field(
        default=2.0, help="Excitation pulse time bandwidth product."
    )
    excitation_duration: int = field(
        default=2000, help="Excitation pulse duration [us]."
    )
    excitation_grad_moment_pre: float = field(
        default=1000.0, help="Excitation pulse slice selective pre-phasing gradient moment [Hz]."
    )
    excitation_grad_rephase_factor: float = field(
        default=1.0, help="Excitation pulse rephase factor. Scales the rephasing slice gradient moment, "
                          "could be used to mitigate slight rephasing deviations."
    )

    # RF - refocus
    refocusing_rf_fa: list[float] | float = field(
        default=140.0, help="Refocusing pulse flip angle(s) [°]"
    )
    refocusing_rf_phase: list[float] | float = field(
        default=0.0 , help="Refocusing pulse phase(s) [°]"
    )
    refocusing_rf_time_bw_prod: float = field(
        default=2.0, help="Refocusing pulse time bandwidth product."
    )
    refocusing_duration: int = field(
        default=2500, help="Refocusing pulse duration [us]"
    )
    refocusing_grad_slice_scale: float = field(
        default=1.5, help="Refocusing pulse slice selective gradient scaling, "
                          "effectively increases slice thickness for refocusing pulses to avoid third arm effect."
    )

    # gradients
    read_grad_spoiling_factor: float = field(
        default=0.5, help="Spoiling gradient strength for readout gradient. Usage after whole readout train (per TR),"
                          " scales the calculated readout gradient moment determined by readout bandwidth."
    )
    grad_moment_slice_spoiling: float = field(
        default=1600.0, help="Spoiling gradient strength for slice selective gradient "
                             "lobes around sl.sel. rf pulses [Hz]."
    )
    grad_moment_slice_spoiling_end: float = field(
        default=2500.0, help="Spoiling gradient strength for slice selective gradient spoiling at "
                             "end of readout train [Hz]."
    )
    # acquisition
    interleaved_slice_acquisition: bool = field(
        default=True, help="Toggle interleaved slice acquisition order."
    )
    sampling_pattern: str = field(
        default="optimized", choices=["interleaved_lines", "random", "grappa", "weighted_random", "optimized"],
        help="Set phase encode sampling pattern between successive echoes."
    )
    use_navs: bool = field(
        default=False, help="Toggle EPI style navigator pulses to be used every TR."
    )
    acq_phase_dir: str = field(
        default="RL", choices=["RL", "PA"], help="Set phase encode direction."
    )
    readout_dwell_us: int = field(
        default=4, help="Readout dwell time [us]."
    )
    oversampling: int = field(
        default=2, help="Readout oversampling factor."
    )
    number_noise_scans: int = field(
        default=10, help="Number of noise scans before starting sequence."
    )

    # timing
    esp: float = field(
        default=8.92, help="Set echo spacing [ms]. "
                           "Shortest possible is calculated and used, except value given here is bigger."
    )
    etl: int = field(
        default=6, help="Echo train length. Defines number of refocusing pulses and echo image readouts."
    )
    tr: float = field(
        default=4500.0, help="Repetition time [ms]."
    )

    # define a bunch of properties (we dont want to serialize those)
    @property
    def resolution_n_read(self) -> int:
        return self.resolution_base  # number of freq encodes

    @property
    def resolution_n_phase(self) -> int:
        # calculate number of phase encodes.
        resolution_n_phase = self.resolution_base * self.resolution_fov_phase / 100
        return int(np.ceil(resolution_n_phase / 2) * 2)

    @property
    def resolution_voxel_size_read(self) -> float:
        return self.resolution_fov_read / self.resolution_base  # in mm

    @property
    def resolution_voxel_size_phase(self) -> float:
        return self.resolution_fov_read / self.resolution_base  # in mm

    @property
    def delta_k_read(self) -> float:
        return 1e3 / self.resolution_fov_read  # cast to m

    @property
    def delta_k_phase(self) -> float:
        return 1e3 / (self.resolution_fov_read * self.resolution_fov_phase / 100.0)  # cast to m

    @property
    def te(self) -> np.ndarray:
        return np.arange(1, self.etl + 1) * self.esp  # in ms echo times

    @property
    def z_extend(self) -> float:
        # there is one gap less than number of slices,
        # in mm
        return self.resolution_slice_thickness * (
                self.resolution_slice_num + self.resolution_slice_gap / 100.0 * (self.resolution_slice_num - 1)
        )

    @property
    def number_outer_lines(self) -> int:
        return int(round(
            (self.resolution_n_phase - self.number_central_lines) / self.acceleration_factor)
        )

    @property
    def dwell(self) -> float:
        return self.readout_dwell_us * 1e-6

    @property
    def acquisition_time(self) -> float:
        return self.resolution_n_read * self.oversampling * self.dwell

    @property
    def bandwidth(self) -> float:
        return 1 / self.acquisition_time

    @property
    def read_grad_fid_spoil_samples(self) ->int:
        # we want to use some FID spoiling of readout gradients,
        # i.e. some additional gradient area before the actual readout, we take this in multiples of readout samples
        # (i.e. multiplied by oversampling and dwell)
        return 48

    @property
    def excitation_rf_rad_fa(self) -> float:
        return self.excitation_rf_fa / 180.0 * np.pi

    @property
    def excitation_rf_rad_phase(self) -> float:
        return self.excitation_rf_phase / 180.0 * np.pi

    @property
    def refocusing_rf_rad_fa(self) -> np.ndarray:
        return np.array(self.refocusing_rf_fa) / 180.0 * np.pi

    @property
    def refocusing_rf_rad_phase(self) -> np.ndarray:
        return np.array(self.refocusing_rf_phase) / 180.0 * np.pi

    def display(self):
        # display via logging
        s = "___ Sequence Parameters ___\n"
        for k, v in self.to_dict().items():
            s += f"\t\t\t{k}:".ljust(30) + f"{v}\n".rjust(55, ".")
        log_module.info(s)

    def __post_init__(self):
        # resolution, number of fe and pe. we want this to be a multiple of 2 for FFT reasoning (have 0 line)
        # on Siemens the number of ADC samples need to be divisible by 4, as of Pulseq docs
        resolution_base = int(np.ceil(self.resolution_base / 4) * 4)
        if np.abs(resolution_base - self.resolution_base) > 1e-6:
            log_module.info(
                f"updating base resolution to {resolution_base} (from {self.resolution_base}) "
                f"due to some FFT and Siemens constraints."
            )
            self.resolution_base = resolution_base
        # if we need to up one freq encode point, we need to update the fov to keep desired voxel resolution
        if np.abs(self.resolution_n_read - self.resolution_base) > 1e-6:
            log_module.info(
                f"updating FOV in read direction from {self.resolution_fov_read:.3f} mm to "
                f"{self.resolution_n_read / self.resolution_base * self.resolution_fov_read:.3f} mm. "
                f"For even frequency encode number")
            self.resolution_fov_read *= self.resolution_n_read / self.resolution_base
            self.resolution_base = self.resolution_n_read

        # we might update the user defined fov phase percentage to the next higher position
        # phase grads - even number of lines. should end up with a 0 line
        if np.abs(self.resolution_n_phase - int(self.resolution_n_phase)) > 0:
            log_module.info(
                f"updating FOV in phase direction from {self.resolution_fov_phase:.2f} % to "
                f"{self.resolution_n_phase / self.resolution_n_read * 100:.2f} % . "
                f"For even phase encode line number")
            self.resolution_fov_phase = self.resolution_n_phase / self.resolution_n_read * 100

        # dwell needs to be on adc raster time, acquisition time is flexible -> leads to small deviations in bandwidth
        # adc raster here hardcoded
        #       note that ADC samples must be on ADC raster time, but the ADC start time must be on RF raster time!
        #       see https://github.com/pulseq/pulseq/blob/master/doc%2Fpulseq_shapes_and_times.pdf for details,
        #       thus set raster time to adc raster, assume us
        # since we shift the adc start about half a sample, according to pulseq info the samples are taken at dwell mitpoint
        # this together makes an even dwell time preferable
        if self.readout_dwell_us % 2 > 0:
            # take next higher
            self.readout_dwell_us += 1
            log_module.warning(
                f"Readout sampling is done at half the dwell time. Thus ADCs are shifted about half Dwell time. "
                f"According to Pulseq, ADCs should start at RF raster time, hence adopting dwell time (from {self.readout_dwell_us -1} to {self.readout_dwell_us} us\n"
                f"If your RF raster time is < 1us you can and need to change this behaviour in code.\n"
                f"Details: https://github.com/pulseq/pulseq/blob/master/doc%2Fpulseq_shapes_and_times.pdf"
            )
        # kf = 1 / (4 * self.bandwidth * adc_raster * self.resolution_n_read * self.oversampling)
        # k = int(np.round(kf)) + 1e-12
        # if np.abs(k - kf) > 1e-7:
        #     bw = self.bandwidth
        #     self.bandwidth = 1 / (4 * k * self.resolution_n_read * self.oversampling * adc_raster)
        #     log_module.info(f"setting dwell time on adc raster -> small bw adoptions (set bw: {bw:.3f}; new bw: {self.bandwidth:.3f})")
        log_module.debug(
            f"\n\t\tBandwidth: {self.bandwidth:.3f} Hz/px;\n"
            f"\t\tReadout time: {self.acquisition_time * 1e3:.1f} ms;\n"
            f"\t\tDwellTime: {self.readout_dwell_us:.1f} us;\n"
            f"\t\tNumber of Freq Encodes: {self.resolution_n_read};¸\n"
            f"\t\tOversampling: {self.oversampling}")
        # make refocusing pulses list of length etl
        if not isinstance(self.refocusing_rf_fa, list):
            self.refocusing_rf_fa = [self.refocusing_rf_fa]
        if not isinstance(self.refocusing_rf_phase, list):
            self.refocusing_rf_phase = [self.refocusing_rf_phase]
        # check if information mismatch
        if self.refocusing_rf_fa.__len__() != self.refocusing_rf_phase.__len__():
            err = f"provide same amount of refocusing pulse angle ({self.refocusing_rf_fa.__len__()}) " \
                  f"and phases ({self.refocusing_rf_phase.__len__()})"
            log_module.error(err)
            raise AttributeError(err)
        # check for phase values
        for l_idx in range(self.refocusing_rf_phase.__len__()):
            while np.abs(self.refocusing_rf_phase[l_idx]) > 180.0:
                self.refocusing_rf_phase[l_idx] = self.refocusing_rf_phase[l_idx] - \
                                                  np.sign(self.refocusing_rf_phase[l_idx]) * 180.0
            while np.abs(self.refocusing_rf_fa[l_idx]) > 180.0:
                self.refocusing_rf_fa[l_idx] = self.refocusing_rf_fa[l_idx] - np.sign(
                    self.refocusing_rf_fa[l_idx]) * 180.0
        while self.refocusing_rf_fa.__len__() < self.etl:
            # fill up list with last value
            self.refocusing_rf_fa.append(self.refocusing_rf_fa[-1])
            self.refocusing_rf_phase.append(self.refocusing_rf_phase[-1])
        # while self.sliceSpoilingMoment.__len__() < self.ETL:
        #     self.sliceSpoilingMoment.append(self.sliceSpoilingMoment[-1])

        # casting
        self.get_voxel_size()
        if self.acq_phase_dir == "PA":
            self.read_dir = 'x'
            self.phase_dir = 'y'
        elif self.acq_phase_dir == "RL":
            self.phase_dir = 'x'
            self.read_dir = 'y'
        else:
            err = 'Unknown Phase direction: chose either PA or RL'
            log_module.error(err)
            raise AttributeError(err)

        # error catches
        if np.any(np.array(self.grad_moment_slice_spoiling) < 1e-7):
            err = f"this implementation needs a spoiling moment supplied: provide spoiling Moment > 0"
            log_module.error(err)
            raise ValueError(err)

    def get_voxel_size(self, write_log: bool = False):
        msg = (
            f"Voxel Size [read, phase, slice] in mm: "
            f"{[self.resolution_voxel_size_read, self.resolution_voxel_size_phase, self.resolution_slice_thickness]}"
        )
        if write_log:
            log_module.info(msg)
        else:
            log_module.debug(msg)
        return self.resolution_voxel_size_read, self.resolution_voxel_size_phase, self.resolution_slice_thickness

    def get_fov(self):
        fov_read = 1e-3 * self.resolution_fov_read
        fov_phase = 1e-3 * self.resolution_fov_read * self.resolution_fov_phase / 100
        fov_slice = self.z_extend * 1e-3
        if self.read_dir == 'x':
            log_module.info(
                f"FOV (xyz) Size [read, phase, slice] in mm: "
                f"[{1e3 * fov_read:.1f}, {1e3 * fov_phase:.1f}, {1e3 * fov_slice:.1f}]")
            return fov_read, fov_phase, fov_slice
        else:
            log_module.info(
                f"FOV (xyz) Size [phase, read, slice] in mm: "
                f"[{1e3 * fov_phase:.1f}, {1e3 * fov_read:.1f}, {1e3 * fov_slice:.1f}]")
            return fov_phase, fov_read, fov_slice



@dataclass
class SystemSpecifications(Serializable):
    b_0: float = 6.98
    max_grad: float = 100.0
    grad_unit: str = "mT/m"
    max_slew: float = 140.0
    slew_unit: str = "T/m/s"
    rise_time: float = 0,
    grad_raster_time: float = 1e-05
    rf_dead_time: float = 100e-6
    rf_raster_time: float = 1e-06
    rf_ringdown_time: float = 3e-05
    adc_dead_time: float = 2e-05
    gamma: float = 42577478.518

    def display(self):
        # display via logging
        s = "___ System Specifications ___\n"
        for k, v in self.to_dict().items():
            s += f"\t\t\t{k}:".ljust(30) + f"{v}\n".rjust(55, ".")
        log_module.info(s)
