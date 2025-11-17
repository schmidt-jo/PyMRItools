import logging
from simple_parsing import field
from dataclasses import dataclass
from pymritools.config import BaseClass


@dataclass
class CoilSensConfig(BaseClass):
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
    coil_dimension: int = field(
        alias="-dim", default=3,
        help="Specify coil dimension along which to compute sensitivity, remember 0 indexing!"
    )
    smoothing_kernel: int = field(
        alias="-k", default=10,
        help="Specify smoothing kernel size for low pass filter."
    )

