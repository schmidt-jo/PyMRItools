from pymritools.config import BaseClass
from dataclasses import dataclass
import logging
from simple_parsing import field
log_module = logging.getLogger(__name__)


@dataclass
class SettingsMPPCA(BaseClass):
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
        default=True, alias="-iimg", help="if input is in image space set to true. "
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


@dataclass
class SettingsMPK(BaseClass):
    """
    Configuration for mp - kspace filter denoising
    """
    in_k_space: str = field(
        alias="-i", default="",
        help="Input k-space .pt file"
    )
    in_noise_scans: str = field(
        alias="-in", default="",
        help="Input noise scans file."
    )
    in_affine: str = field(
        alias="-ia", default="",
        help="input affine matrix, necessary if input file is .pt, optional if .nii"
    )
    in_sampling_mask: str = field(
        alias="-im", default="",
        help="(optional) input sampling mask to reduce computational cost by computing only sampled pe/slice lines. "
             "Can be deduced from k-space itself."
    )

    file_prefix: str = field(
        default="d", alias="-fp",
        help=f"Output file prefix appended to name after denoising"
    )
    # flags & vars
    batch_size: int = field(
        alias="-b", default=200,
        help=f"Batch size for processing"
    )
    noise_histogram_depth: int = field(
        alias="-nh", default=100,
        help="Sampling depth at which to sample the noise histogram axes for the noise scans."
    )
    noise_mp_threshold: float = field(
        alias="-nth", default=0.3,
        help="The MP distribution estimated from the noise is normalized, then clamped to 1 if exceeding this threshold. "
             "This way effectively each singular value below this threshold in the mp distribution is hard thresholded. "
             "Above a smoothed inverted distribution curve is used"
    )
    noise_mp_stretch: float = field(
        alias="-ns", default=1.0,
        help="The estimated noise distribution is stretched by this factor above its bandwidth. "
             "That way singular values above the distribution bandwidth will get affected by the filter"
    )


@dataclass
class SettingsKLC(BaseClass):
    """
    Configuration for klc - pca - kspace filter denoising
    """
    in_k_space: str = field(
        alias="-i", default="",
        help="Input k-space .pt file"
    )
    in_noise_scans: str = field(
        alias="-in", default="",
        help="Input noise scans file."
    )
    in_affine: str = field(
        alias="-ia", default="",
        help="input affine matrix, necessary if input file is .pt, optional if .nii"
    )

    file_prefix: str = field(
        default="d", alias="-fp",
        help=f"Output file prefix appended to name after denoising"
    )
    # flags & vars
    batch_size: int = field(
        alias="-b", default=200,
        help=f"Batch size for processing"
    )
    line_patch_size: int = field(
        alias="-lp", default=0,
        help=f"Patch size along the readout line (pca matrices formed from these patches and channel dimension). "
             f"If 0, channel dim will be matched (default)."
    )
    noise_dist_area_threshold: float = field(
        alias="-ndth", default=0.85,
        help=f"Threshold area. We build a smooth function downweighting all eigenvalues below this threshold. "
             f"Its defined as the area under the eigenvalue distribution curve of the noise. "
             f"Eigenvalue within this area will be attributed to noise in the filtering process."
    )
    ev_cutoff_percent_range: float = field(
        alias="-ecpr", default=15.0,
        help=f"Around the threshold the eigenvalues are weighted from 0 (below) to 1 (above) "
             f"to perform the noise filtering, using a smoothstep function. "
             f"This parameter defines the width in percentage of the threshold of the step."
    )
