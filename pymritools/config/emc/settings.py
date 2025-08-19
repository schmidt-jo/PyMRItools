import logging
from dataclasses import dataclass
from simple_parsing import field
from pymritools.config import BaseClass
log_module = logging.getLogger(__name__)


@dataclass
class SimulationSettings(BaseClass):
    """
    EMC - Bloch equation simulation settings
    """
    # files and config
    emc_params_file: str = field(
        alias="-emc", default="./examples/simulation/emc/emc_params.json",
        help="provide sequence event parameters"
    )
    kernel_file: str = field(
        alias="-kernels", default="./examples/simulation/kernels/kernels",
        help="provide file to kernel dict., i.e. named sequence event blocks"
    )
    te_file: str = field(
        alias="-te", default="",
        help="provide list of tes"
    )
    pulse_file: str = field(
        alias="-pul", default="./examples/simulation/emc/pulse_pypulseq_default_gauss.pkl",
        help="separate pulse file to pulse class object"
    )
    database_name: str = field(
        alias="-db", default="database_test.pkl",
        help="set filename of database"
    )

    t1_list: list = field(
        alias="-var_t1", default_factory=lambda: [1.5],
        help="T1 to simulate [s]. List of all values."
    )
    t2_list: list = field(
        alias="-var_t2", default_factory=lambda: [[25, 30, 0.5], [30, 35, 1]],
        help="T2 to simulate [ms], List of ranges [(start_1, end_1, step_1), ..., (start_n, end_n, step_n)]"
    )
    b1_list: list = field(
        alias="-var_b1", default_factory=lambda: [0.6, 1.6, 0.1],
        help="B1 to simulate, List of ranges [(start_1, end_1, step_1), ..., (start_n, end_n, step_n)]."
    )
    b0_list: list = field(
        alias="-var_b0", default_factory=lambda: [[0, 1, 2]],
        help="B0 to simulate, List of ranges [(start_1, end_1, step_1), ..., (start_n, end_n, step_n)]."
    )

    # options
    signal_fourier_sampling: bool = field(
        alias="-sfs", default=False,
        help="set signal echo processing -> this enables sampling the signal over the 1d slice dimension, "
             "substituting the readout and using identical readout time etc.. "
             "When turned off the spin contributions are summed across the profile"
    )
    resample_pulse_to_dt_us: float = field(
        alias="-rptdt", default=5.0,
        help="resample pulse to lower number (duration over dt) for more efficient computations"
    )


@dataclass
class FitSettings(BaseClass):
    """
    Configuration for dictionary matching of simulated databases
    """
    input_data: str = field(
        alias="-i", default="",
        help="Path to input data (.nii) file."
    )
    input_database: str = field(
        alias="-db", default="",
        help="Path to input database (.pkl) file."
    )
    input_b1: str = field(
        alias="-b1", default="",
        help="Path to input B1+ map (.nii) file if available."
    )
    input_b0: str = field(
        alias="-b0", default="",
        help="Path to input B0 map (.nii) if available (only used in MEGESSE fitting)."
    )
    input_affine: str = field(
        alias="-ia", default="",
        help="(optional) Input affine for .pt data input. "
             "If input data is .pt data and no affine is given, the identity matrix is used. "
             "Might lead to misalignment when loading into nifti viewers."
    )
    out_name: str = field(
        alias="-on", default="",
        help="Optional output filename."
    )
    save_name_prefix: str = field(
        alias="-pre", default="",
        help="Optional name prefix for output data"
    )
    # flags & vars
    batch_size: int = field(
        alias="-bs", default=3000,
        help="Set batch size for batched processing of input signal."
    )
    process_slice: bool = field(
        alias="-ps", default=False,
        help="Process middle single slice only to reduce processing time, eg. for debugging or qa."
    )
    rsos_channel_combine: bool = field(
        alias="-rsos", default=False,
        help="Process combined magnitude images."
             "Otherwise the fitting is done channel wise and a weighted averaging based on the goodness fo fit is used for combination"
    )
    input_in_image_space: bool = field(
        alias="-iimg", default=False,
        help="if False, toggle FFT to get input to image space first for k-space input."
    )
    low_rank_regularisation: bool = field(
        alias="-lr", default=True,
        help="For noisy channel wise matching, we can regularise the matching by using low-rank approximations in small neighborhoods in the fitting."
    )


