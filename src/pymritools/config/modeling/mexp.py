import logging
from dataclasses import dataclass
from simple_parsing import field

from pymritools.config import BaseClass

log_module = logging.getLogger(__name__)


@dataclass
class MEXPSettings(BaseClass):
    input_data: str = field(
        alias="-i", default="",
        help="Filename of the input data file. (provide .pt or .nii files)",
    )
    input_affine: str = field(
        alias="-ia", default="",
        help="Filename of the input affine file. (needed if input data is .pt file)",
    )
    input_in_img_space: bool = field(
        alias="-ii", default=True,
        help="Specify if input is in image space, otherwise provide k-space data"
    )
    echo_times: list = field(
        alias="-te", default_factory=[0],
        help="list of echo times in seconds."
    )
