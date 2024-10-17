import logging
import pathlib as plib

import torch
import numpy as np

from pymritools.config.seqprog import RD, Sampling, PulseqParameters2D
from pymritools.config import setup_program_logging, setup_parser
from pymritools.seqprog.rawdata.load_fns import load_pulseq_rd
from pymritools.utils import torch_save, fft, root_sum_of_squares, nifti_save, HidePrints
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
    if ".dat" not in path_to_file.suffixes:
        err = "File not recognized as .dat raw data file."
        log_module.error(err)
        raise ValueError(err)
    # ToDo: set device - have to optimize to fit gpu, for now too big
    # if RD.use_gpu and torch.cuda.is_available():
    #     device = torch.device(f"cuda:{RD.gpu_device}")
    # else:
    device = torch.device("cpu")
    with HidePrints():
        twix = twixtools.read_twix(path_to_file.as_posix(), parse_geometry=True, verbose=True, include_scans=-1)[0]
    geometry = twix["geometry"]
    data_mdbs = twix["mdb"]
    hdr = twix["hdr"]
    log_module.info("Loading RD")
    k_space, k_sampling_mask, aff = load_pulseq_rd(
        pulseq_config=pulseq, sampling_config=sampling,
        data_mdbs=data_mdbs, geometry=geometry, hdr=hdr,
        device=device
    )

    log_module.info("Saving")
    # output path
    path_out = plib.Path(config.out_path).absolute()

    if config.visualize:
        # transform into image
        img = fft(k_space, img_to_k=False, axes=(0, 1))
        for i in np.random.randint(low=0, high=img.shape[3], size=(3,)):
            nifti_save(data=np.abs(img[:, :, :, i]), img_aff=aff, path_to_dir=path_out, file_name=f"naive_recon_mag_ch-{i}")
            nifti_save(data=np.angle(img[:, :, :, i]), img_aff=aff, path_to_dir=path_out, file_name=f"naive_recon_phase_ch-{i}")
        # do rSoS
        img = root_sum_of_squares(img, dim_channel=-2)
        # nifti save
        nifti_save(data=img, img_aff=aff, path_to_dir=path_out, file_name="naive_rsos_recon")
        nifti_save(data=k_sampling_mask.astype(float), img_aff=aff, path_to_dir=path_out, file_name="sampling_pattern")

    # save as torch tensor for recon
    torch_save(k_space, path_out, "k_space")
    torch_save(k_sampling_mask, path_out, "k_sampling_mask")
    torch_save(aff, path_out, "affine")


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
