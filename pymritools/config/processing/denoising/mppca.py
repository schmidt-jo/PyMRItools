from pymritools.config.base import BaseClass
from dataclasses import dataclass
import logging
from simple_parsing import field
log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    """
    Configuration for mppca denoising
    """
    in_path: str = field(
        alias="-i", default="./examples/processing/semc_r0p9_echo_mag.nii.gz",
        help="set filepath to .nii or .pt file"
    )
    in_affine: str = field(
        alias="-ia", default="",
        help="input affine matrix, necessary if input file is .pt, optional if .nii"
    )
    file_prefix: str = field(
        default="d", alias="-fp",
        help=f"Output file prefix appended to name after denoising / debiasing"
    )
    # flags
    normalize: bool = field(
        default=False, alias="-n", help="(optional), normalize data (across t dimension) to max 1 before pca"
    )
    input_image_data: bool = field(
        default=False, alias="-iimg", help="if input is in image space set to true. "
                                           "Otherwise input is assumed to be k-space data"
    )
    noise_bias_correction: bool = field(
        default=False, alias="-nbc",
        help="(optional) noise bias correction "
             "using stationary or non stationary noise estimates and "
             "assuming non-central chi noise distribution."
    )
    # vars
    fixed_p: int = field(
        default=0, alias="-p", help="(optional) fix the number of singular values to keep in patch."
                                    "For (default) 0 the number is computed per patch from the MP inequality."
    )
    noise_bias_mask: str = field(
        default="", alias="-nbm", help="input noise mask for noise statistics estimation if bias correction is set."
    )
