"""
Holds all necessary configuration parameters for EMC sequence simulations and dictionary matching.
_____
Jochen Schmidt, 26.09.2024
"""
from dataclasses import dataclass
from simple_parsing.helpers import Serializable
from simple_parsing import field
from .settings import SimulationSettings
import numpy as np
import torch
import logging
from scipy import stats
from scipy.constants import physical_constants
log_module = logging.getLogger(__name__)


@dataclass
class Parameters(Serializable):
    """
    EMC - Bloch equation simulation parameters.
    """
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

    tes: list = field(
        default_factory=lambda: [0.0],
        help="list of echo times [s]"
    )

    # sample parameter, spatial resolution
    sample_number: int = field(
        alias="var_sn", default=1000,
        help="no of sampling points along slice profile"
    )
    length_z: float = field(
        alias="var_lz", default=0.005,
        help="[m] length extension of z-axis spanned by sample -> total length 2*lengthZ (-:+)"
    )
    acquisition_number: int = field(
        alias="var_an", default=50,
        help="number of bins across slice sample -> effectively sets spatial resolution; "
             "resolution = 2 * lengthZ / acquisitionNumber"
    )

    @property
    def gamma_hz(self) -> float:
        return physical_constants["proton gyromag. ratio in MHz/T"][0] * 1e6

    @property
    def gamma_pi(self) -> float:
        return self.gamma_hz * 2 * np.pi

    @property
    def duration_acquisition(self) -> float:
        return 1e6 / self.bw  # [us]

    # set sample resolution
    def set_spatial_sample_resolution(self, sample_length_z: float, sample_number_of_acq_bins: int):
        self.length_z = sample_length_z
        self.acquisition_number = sample_number_of_acq_bins

    @property
    def gradient_acquisition(self) -> float:
        if self.acquisition_number is None or self.length_z is None:
            err = (f"sample resolution and length not set (acq.no.: {self.acquisition_number}, length: {self.length_z}) "
                   f"but needed for simulation.")
            log_module.error(err)
            ValueError(err)
        # gradient area = deltaK * n = 1/FOV * num_acquisition
        grad_area = 1 / (2 * self.length_z) * self.acquisition_number
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


class SimulationData:
    """
    Setup and allocate tensors to carry data through the simulation and set up the device (i.e. GPU acceleration)
    """
    def __init__(self, params: Parameters, settings: SimulationSettings, device: torch.device = torch.device("cpu")):
        log_module.debug("\t\tSetup Simulation Data")
        self.device: torch.device = device
        # setup axes in [m] positions along z axes
        self.sample_axis: torch.Tensor = torch.linspace(
            -params.length_z, params.length_z, params.sample_number
        ).to(self.device)
        sample = torch.from_numpy(
            stats.gennorm(24).pdf(self.sample_axis / params.length_z * 1.1) + 1e-6
        )
        # build bulk magnetization along axes positions
        self.sample: torch.Tensor = torch.divide(sample, torch.max(sample))

        # set values with some error catches
        # t1
        if isinstance(settings.t1_list, list):
            t1_vals = torch.tensor(settings.t1_list, device=self.device)
        else:
            t1_vals = torch.tensor([settings.t1_list], dtype=torch.float32, device=self.device)
        self.t1_vals: torch.Tensor = t1_vals
        self.num_t1s: int = self.t1_vals.shape[0]
        # b1
        self.b1_vals: torch.Tensor = self._build_tensors_from_list_of_lists_args(settings.b1_list).to(
            self.device)
        self.num_b1s: int = self.b1_vals.shape[0]
        # t2
        array_t2 = self._build_tensors_from_list_of_lists_args(settings.t2_list)
        array_t2 /= 1000.0  # cast to s
        self.t2_vals: torch.Tensor = array_t2.to(self.device)
        self.num_t2s: int = self.t2_vals.shape[0]
        # sim info
        # set total number of simulated curves
        num_curves = self.num_t1s * self.num_t2s * self.num_b1s
        log_module.info(f"\t\t- total number of entries to simulate: {num_curves}")
        self.total_num_sim: int = num_curves

        # set magnetization vector and initialize magnetization, insert axes for [t1, t2, b1] number of simulations
        m_init = torch.zeros((params.sample_number, 4))
        m_init[:, 2] = self.sample
        m_init[:, 3] = self.sample
        # save initial state
        self.m_init: torch.Tensor = m_init[None, None, None].to(self.device)
        # set initial state as magnetization propagation tensor, this one is used
        # to iteratively calculate the magnetization sate along the axis
        self.magnetization_propagation: torch.tensor = self.m_init.clone()

        self.gamma = torch.tensor(params.gamma_hz, device=self.device)

        # signal tensor is supposed to hold all acquisition points for all reads
        self.signal_tensor: torch.Tensor = torch.zeros(
            (self.num_t1s, self.num_t2s, self.num_b1s, params.etl, params.acquisition_number),
            dtype=torch.complex128, device=self.device
        )

        # allocate magnitude and phase results
        # set emc data tensor -> dims: [t1s, t2s, b1s, ETL]
        self.signal_mag: torch.Tensor = torch.zeros(
            (self.num_t1s, self.num_t2s, self.num_b1s, params.etl),
            device=self.device
        )
        self.signal_phase: torch.Tensor = torch.zeros(
            (self.num_t1s, self.num_t2s, self.num_b1s, params.etl),
            device=self.device
        )

    @property
    def complete_param_list(self) -> list:
        return [(t1, t2, b1) for t1 in self.t1_vals
                for t2 in self.t2_vals for b1 in self.b1_vals]


    @staticmethod
    def _build_tensors_from_list_of_lists_args(val_list) -> torch.tensor:
        """
        We use a list of values or value ranges in the configuration (CLI or config file) and
        want to translate this to a tensor of all values
        :param val_list: list of values or list of tuples / lists of value ranges
        :return: tensor of all values to simulate
        """
        array = []
        if isinstance(val_list, list):
            for item in val_list:
                if isinstance(item, str):
                    item = [float(i) for i in item[1:-1].split(',')]
                if isinstance(item, int):
                    item = float(item)
                if isinstance(item, float):
                    array.append(item)
                else:
                    array.extend(torch.arange(*item).tolist())
        else:
            array = [val_list]
        return torch.tensor(array, dtype=torch.float32)

    def _check_args(self):
        # sanity checks
        if torch.max(self.t2_vals) > torch.min(self.t1_vals):
            err = 'T1 T2 mismatch (T2 > T1)'
            log_module.error(err)
            raise AttributeError(err)
        if torch.max(self.t2_vals) < 1e-4:
            err = 'T2 value range exceeded, make sure to post T2 in ms'
            log_module.error(err)
            raise AttributeError(err)

    def set_device(self):
        """
        push all tensors to given torch device
        :param device: torch device (cpu or gpu)
        :return: nothing
        """
        for _, value in vars(self).items():
            if torch.is_tensor(value):
                value.to(self.device)
