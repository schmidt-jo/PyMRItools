import logging
from dataclasses import dataclass
from simple_parsing import field
from pymritools.config.base import BaseClass
log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    """
    EMC - Bloch equation simulation settings
    """
    # files and config
    emc_params_file: str = field(
        alias="-emc", default="./examples/simulation/emc_params.json",
        help="provide sequence event parameters"
    )
    pulse_file: str = field(
        alias="-pul", default="./examples/simulation/pulse_pypulseq_default_gauss.pkl",
        help="separate pulse file to pulse class object"
    )
    save_path: str = field(
        alias="-s", default="./examples/simulation/results",
        help="set path to save database and used config"
    )
    database_name: str = field(
        alias="-db", default="database_test.pkl",
        help="set filename of database"
    )
    sim_type: str = field(
        alias="-t",
        default="mese_balanced_read",
        choices=["mese_siemens", "mese_balanced_read", "megesse", "fid", "single"],
        help= "set simulation type"
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

    # flags
    visualize: bool = field(
        alias="-v", default=True,
        help="visualize pulse profiles and sequence scheme"
    )
    debug: bool = field(
        alias="-d", default=False,
        help="toggle debugging mode, and logging debug level"
    )
    use_gpu: bool = field(
        alias="-gpu", default=False,
        help="init gpu"
    )

    gpu_device: int = field(
        alias="-gpud", default=0,
        help="(optional) set gpu device if multiple are available"
    )
