import logging
import pathlib as plib

import numpy as np
from dataclasses import dataclass
from simple_parsing import field
from scipy.ndimage import gaussian_filter

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.basic import BaseClass
from pymritools.utils import nifti_load, nifti_save

log_module = logging.getLogger(__name__)


@dataclass
class Settings(BaseClass):
    input_file: str = field(
        alias="-i", default="",
        help="Input file path to combined AFI echo images."
    )
    flip_angle: float = field(
        alias="-fa", default=60.0,
        help="Flip angle in degrees."
    )
    ratio_tr2_tr1: int = field(
        alias="-n", default=5,
        help="Ratio of TR2 / TR1."
    )
    smoothing_kernel: float = field(
        alias="-s", default=3,
    )


def afi_b1(settings: Settings):
    # load file
    b1_data, b1_img = nifti_load(settings.input_file)
    # calculate ratio
    r = np.divide(
        b1_data[..., 1], b1_data[..., 0],
        where=np.abs(b1_data[..., 0]) > 1e-9, out=np.zeros_like(b1_data[..., 1])
    )
    arg = np.divide(
            r * settings.ratio_tr2_tr1 - 1, settings.ratio_tr2_tr1 - r,
            where=np.abs(settings.ratio_tr2_tr1 - r) > 1e-9, out=np.zeros_like(r)
        )
    alpha = np.arccos(np.clip(arg, -1, 1))
    alpha *= 180 / np.pi
    alpha_filtered = gaussian_filter(alpha, sigma=settings.smoothing_kernel, axes=(0, 1, 2))

    b1 = alpha / settings.flip_angle * 100
    b1_f = alpha_filtered / settings.flip_angle * 100

    # save
    nifti_save(data=b1, img_aff=b1_img, path_to_dir=settings.out_path, file_name="b1_afi_unfilt")
    nifti_save(data=b1_f, img_aff=b1_img, path_to_dir=settings.out_path, file_name="b1_afi")
    nifti_save(data=b1_data[..., 0], img_aff=b1_img, path_to_dir=settings.out_path, file_name="b1_afi_ref")


def main():
    setup_program_logging(name="AFI B1 creation")
    parser, args = setup_parser(prog_name="AFI B1 creation", dict_config_dataclasses={"settings": Settings})
    # get cli args
    settings = Settings.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        afi_b1(settings=settings)
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()

