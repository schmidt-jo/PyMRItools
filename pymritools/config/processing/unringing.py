import logging
from dataclasses import dataclass
from simple_parsing import field
from pymritools.config.base import BaseClass

log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    input_file: str = field(
        alias="-i", default="",
        help="Input Nifti file of data to unring."
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
