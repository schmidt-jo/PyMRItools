import logging
from pymritools.config.seqprog import RD, Sampling, PulseqParameters2D
from pymritools.config import setup_program_logging, setup_parser
from pymritools.seqprog.rawdata.load_fns import load_pulseq_rd
from pymritools.utils import torch_save, fft, root_sum_of_squares, nifti_save

import pathlib as plib
import twixtools
log_module = logging.getLogger(__name__)


def rd_to_torch(config: RD):
    # load sampling configuration
    path_to_file = plib.Path(config.input_sample_config).absolute()
    if not path_to_file.exists():
        err = f"File {path_to_file.as_posix()} does not exist."
        log_module.error(err)
        raise FileNotFoundError(err)
    sampling = Sampling.load(path_to_file.as_posix())

    # load sequence config
    path_to_file = plib.Path(config.input_sequence_config).absolute()
    if not path_to_file.exists():
        err = f"File {path_to_file.as_posix()} does not exist."
        log_module.error(err)
        raise FileNotFoundError(err)
    pulseq = PulseqParameters2D.load(path_to_file.as_posix())

    # load data
    path_to_file = plib.Path(config.input_file).absolute()
    if not path_to_file.exists():
        err = f"File {path_to_file.as_posix()} does not exist."
        log_module.error(err)
        raise FileNotFoundError(err)
    log_module.info(f"Loading raw data file: {path_to_file.as_posix()}")
    if not ".dat" in path_to_file.suffixes:
        err = "File not recognized as .dat raw data file."
        log_module.error(err)
        raise ValueError(err)
    twix = twixtools.read_twix(path_to_file.as_posix(), parse_geometry=True, verbose=True, include_scans=-1)[0]
    geometry = twix["geometry"]
    data_mdbs = twix["mdb"]
    hdr = twix["hdr"]
    log_module.info("Loading RD")
    k_space, k_sampling_mask, aff = load_pulseq_rd(
        pulseq_config=pulseq, sampling_config=sampling,
        data_mdbs=data_mdbs, geometry=geometry, hdr=hdr
    )

    log_module.info("Saving")
    # output path
    path_out = plib.Path(config.out_path).absolute()

    # save as np
    torch_save(k_space, path_out, "k_space")
    torch_save(k_sampling_mask, path_out, "k_sampling_mask")
    torch_save(aff, path_out, "affine")

    if config.visualize:
        # transform into image
        img = fft(k_space, img_to_k=False, axes=(0, 1))
        # do rSoS
        img = root_sum_of_squares(img, dim_channel=-2)
        # niftii save
        nifti_save(data=img, img_aff=aff, path_to_dir=path_out, file_name="naive_rsos_recon")


def main():
    # setup logging
    setup_program_logging(name="Raw Data to torch", level=logging.INFO)

    # setup parser
    parser, args = setup_parser(prog_name="Raw Data to torch", dict_config_dataclasses={"settings": RD})

    # get config
    rd_config = RD.from_cli(args=args.settings, parser=parser)
    rd_config.display()

    try:
        rd_to_torch(config=rd_config)
    except Exception as e:
        parser.print_help()
        logging.exception(e)


if __name__ == '__main__':
    main()