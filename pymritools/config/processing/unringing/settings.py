import logging
from dataclasses import dataclass
from simple_parsing import field
from pymritools.config.base import BaseClass
import pathlib as plib

log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    config_file: str = field(
        alias="-c", default="",
        help="Input configuration file (.json) covering entries to this Settings object."
    )
    input_nifti_file: str = field(
        alias="-i",
        help="Input Nifti file of data to unring."
    )
    output_path: str = field(
        alias="-o",
        help="Output path for processed data."
    )
    # vars
    num_shifts_per_voxel: int = field(
        alias="-m", default=100,
        help="The number of shifted images to be computed per voxel. Default is 100."
    )
    voxel_neighborhood_size: int = field(
        alias="-k", default=3,
        help="The size of the neighborhood around each voxel for which to find the optimal shifted image. Default is 3."
    )
    # flags
    visualize: bool = field(
        alias="-v", default=False,
        help="Visualize results."
    )
    debug: bool = field(
        alias="-d", default=False,
        help="Toggle debugging mode, and logging debug level."
    )
    use_gpu: bool = field(
        alias="-gpu", default=False,
        help="Use GPU acceleration if available."
    )
