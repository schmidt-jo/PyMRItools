import logging
from dataclasses import dataclass
import polars as pl
from simple_parsing import field
from simple_parsing.helpers import Serializable

log_module = logging.getLogger(__name__)


@dataclass
class Settings(Serializable):
    """
    EMC - Bloch equation simulation settings
    """
    # files and config
    config_file: str = field(
        alias="-c", default="../example/simulation/emc_settings.json",
        help=" provide Configuration file (.json)"
    )
    emc_params_file: str = field(
        alias="-emc", default="../example/simulation/emc_params.json",
        help="provide sequence event parameters"
    )
    pulse_file: str = field(
        alias="-opul", default="/example/simulation/pulse_pypulseq_default_gauss.pkl",
        help="separate pulse file to pulse class object"
    )
    save_path: str = field(
        alias="-s", default="../example/simulation/results",
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

    @property
    def complete_param_list(self) -> list:
        return [(t1, t2, b1) for t1 in self.t1_list
                for t2 in self.t2_list for b1 in self.b1_list]

    @property
    def  total_num_sim(self) -> int:
        return len(self.complete_param_list)

    def display(self):
        # display via logging
        s = "___ Config ___\n"
        for k, v in self.to_dict().items():
            s += f"\t\t\t{k}: {v}\n"
        log_module.info(s)
