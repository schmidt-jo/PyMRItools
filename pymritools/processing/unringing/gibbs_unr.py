"""
Gibbs unringing algorithm based on doi/10.1002/mrm.26054
Kellner et al. 2015 MRM
_____
Implementation, Jochen Schmidt, 26.09.2024

"""
import pathlib as plib

import torch

from pymritools.processing.unringing import gibbs_unring_nd
from pymritools.config.processing import GibbsUnringingSettings
from pymritools.utils import nifti_load, nifti_save
from pymritools.config import setup_program_logging, setup_parser
import logging
log_module = logging.getLogger(__name__)


def main():
    setup_program_logging(name="Gibbs Unringing", level=logging.INFO)

    # setup argument parser
    parser, prog_args = setup_parser(
        prog_name="Gibbs Unringing",
        dict_config_dataclasses={"settings": GibbsUnringingSettings}
    )

    settings = GibbsUnringingSettings.from_cli(args=prog_args.settings, parser=parser)
    settings.display()

    try:
        # prepare output path
        path_out = plib.Path(settings.out_path).absolute()
        path_out.mkdir(exist_ok=True, parents=True)

        # load in data
        path_in = plib.Path(settings.input_file).absolute()
        input_data, input_img = nifti_load(settings.input_file)


        data_unring = gibbs_unring_nd(
            image_data_nd=torch.from_numpy(input_data), visualize=False, gpu=True,
            m=settings.num_shifts_per_voxel, k=settings.voxel_neighborhood_size
        )

        # save data
        nifti_save(
            data=data_unring, img_aff=input_img,
            path_to_dir=settings.out_path, file_name=f"ur_{path_in.stem}"
        )
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    main()
