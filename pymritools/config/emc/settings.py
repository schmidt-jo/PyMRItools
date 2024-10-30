import logging
from dataclasses import dataclass
from simple_parsing import field
from pymritools.config.base import BaseClass
log_module = logging.getLogger(__name__)


@dataclass
class SimulationSettings(BaseClass):
    """
    EMC - Bloch equation simulation settings
    """
    # files and config
    # emc_params_file: str = field(
    #     alias="-emc", default="./examples/simulation/emc/emc_params.json",
    #     help="provide sequence event parameters"
    # )
    kernel_file: str = field(
        alias="-kernels", default="./examples/simulation/kernels/kernels",
        help="provide file to kernel dict., i.e. named sequence event blocks"
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
        alias="var_t1", default_factory=lambda: [1.5],
        help="T1 to simulate [s]. List of all values."
    )
    t2_list: list = field(
        alias="var_t2", default_factory=lambda: [[25, 30, 0.5], [30, 35, 1]],
        help="T2 to simulate [ms], List of ranges [(start_1, end_1, step_1), ..., (start_n, end_n, step_n)]"
    )
    b1_list: list = field(
        alias="var_b1", default_factory=lambda: [0.6, 1.0],
        help="B1 to simulate/ List of all values."
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
        help="Path to input b1 map (.nii) file."
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


