import logging
from pymritools.config import BaseClass
from dataclasses import dataclass
from simple_parsing import field
log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    """
    Configuration for PyLORAKS reconstruction
    """
    input_k_space: str = field(
        alias="-ik", default="./examples/raw_data/results/k_space.pt",
        help="set filepath to .pt file"
    )
    in_affine: str = field(
        alias="-ia", default="./examples/raw_data/results/k_space.pt",
        help="input affine matrix, necessary if input file is .pt to output .nii file."
    )
    in_sampling_mask: str = field(
        alias="-is", default="",
        help=f"(Optional) Input sampling mask for reconstruction masking sampled voxels in the input."
             f" If not given it will be deduced from the input."
    )
