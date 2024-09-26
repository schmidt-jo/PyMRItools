"""
Holds all necessary configuration parameters for EMC sequence simulations and dictionary matching.
_____
Jochen Schmidt, 26.09.2024
"""
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from simple_parsing import field
import numpy as np
import logging
log_module = logging.getLogger(__name__)


@dataclass
class Parameters(Serializable):
    """
    EMC - Bloch equation simulation parameters.
    """
    # global parameter gamma [Hz/t]
    gamma_hz: float = 42577478.518

    # echo train length
    etl: int = 16
    # echo spacing [ms]
    esp: float = 9.0
    # bandwidth [Hz/px]
    bw: float = 349
    # gradient mode

    # Excitation, Flip Angle [°]
    excitation_angle: float = 90.0
    # Excitation, Phase [°]
    excitation_phase: float = 90.0
    # Excitation, gradient if rectangular/trapezoid [mt/m]
    gradient_excitation: float = -18.5
    # Excitation, duration of pulse [us]
    duration_excitation: float = 2560.0

    gradient_excitation_rephase: float = -10.51  # [mT/m], rephase
    duration_excitation_rephase: float = 1080.0  # [us], rephase

    # Refocussing, Flip Angle [°]
    refocus_angle: list = field(default_factory=lambda: [140.0])
    # Refocussing, Phase [°]
    refocus_phase: list = field(default_factory=lambda: [0.0])
    # Refocussing, gradient strength if rectangular/trapezoid [mt/m]
    gradient_refocus: float = -36.2
    # Refocussing, duration of pulse [us]
    duration_refocus: float = 3584.0

    gradient_crush: float = -38.7  # [mT/m], crusher
    duration_crush: float = 1000.0  # [us], crusher

    # Verse - assuming the respective gradient specified above is the middle / main gradient,
    # only specify lobes and lobe timing
    gradient_excitation_verse_lobes: float = 0.0
    duration_excitation_verse_lobes: float = 0.0
    gradient_refocus_verse_lobes: float = 0.0
    duration_refocus_verse_lobes: float = 0.0
    # echo times
    tes: list = field(default_factory=lambda: [0.0])

    # For serialization, we don't want to expose some of the attributes to the config API
    _length_z: float = None
    _acq_num: int = None

    @property
    def gamma_pi(self) -> float:
        return self.gamma_hz * 2 * np.pi

    @property
    def duration_acquisition(self) -> float:
        return 1e6 / self.bw  # [us]

    @property
    def length_z(self) -> float:
        return self._length_z

    @property
    def acq_number(self) -> int:
        return self._acq_num

    # set sample resolution
    def set_spatial_sample_resolution(self, sample_length_z: float, sample_number_of_acq_bins: int):
        self._length_z = sample_length_z
        self._acq_num = sample_number_of_acq_bins

    @property
    def gradient_acquisition(self) -> float:
        if self.acq_number is None or self.length_z is None:
            err = (f"sample resolution and length not set (acq.no.: {self.acq_number}, length: {self.length_z}) "
                   f"but needed for simulation.")
            log_module.error(err)
            ValueError(err)
        # gradient area = deltaK * n = 1/FOV * num_acquisition
        grad_area = 1 / (2 * self.length_z) * self.acq_number
        # grad_area in 1/m -> / gamma for T/m
        grad_amp = grad_area / self.gamma_hz / self.duration_acquisition * 1e6  # cast to s
        return - grad_amp * 1e3  # cast to mT

    def __post_init__(self):
        self._check_refocus_settings()

    def _check_refocus_settings(self):
        if self.refocus_phase.__len__() != self.refocus_angle.__len__():
            err = f"provide same amount of refocusing pulse angle ({self.refocus_angle.__len__()}) " \
                  f"and phases ({self.refocus_phase.__len__()})"
            log_module.error(err)
            raise AttributeError(err)
        # check for phase values
        for l_idx in range(self.refocus_phase.__len__()):
            while abs(self.refocus_phase[l_idx]) > 180.0:
                self.refocus_phase[l_idx] = self.refocus_phase[l_idx] - np.sign(self.refocus_phase[l_idx]) * 180.0
            while abs(self.refocus_angle[l_idx]) > 180.0:
                self.refocus_angle[l_idx] = self.refocus_angle[l_idx] - np.sign(self.refocus_angle[l_idx]) * 180.0
        while self.refocus_angle.__len__() < self.etl:
            # fill up list with last value
            self.refocus_angle.append(self.refocus_angle[-1])
            self.refocus_phase.append(self.refocus_phase[-1])
