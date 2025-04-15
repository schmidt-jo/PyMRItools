import logging
import pathlib as plib

import torch
import tqdm
import numpy as np

from pymritools.config.seqprog import RD, Sampling, PulseqParameters2D
from pymritools.config import setup_program_logging, setup_parser
from pymritools.seqprog.rawdata.load_fns import load_pulseq_rd, load_siemens_rd
from pymritools.utils import torch_save, ifft, root_sum_of_squares, nifti_save, HidePrints
import twixtools

log_module = logging.getLogger(__name__)


def pulseq_rd_to_torch(config: RD):
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
    k_space, k_sampling_mask, aff, noise_scans, echoes_bu, echoes_bd = load_pulseq_rd(
        pulseq_config=pulseq, sampling_config=sampling,
        data_mdbs=data_mdbs, geometry=geometry, hdr=hdr,
        use_gpu=config.use_gpu, gpu_device=config.gpu_device,
        remove_os=config.remove_os
    )

    log_module.info("Saving")
    # output path
    path_out = plib.Path(config.out_path).absolute()

    if config.visualize:
        # transform into image
        img = np.zeros_like(k_space)
        for i in tqdm.trange(img.shape[2], desc="FFT"):
            img[:, :, i] = ifft(k_space[:, :, i], dims=(0, 1))
        if config.debug:
            for i in np.random.randint(low=0, high=img.shape[3], size=(3,)):
                nifti_save(data=np.abs(img[:, :, :, i]), img_aff=aff, path_to_dir=path_out,
                           file_name=f"naive_recon_mag_ch-{i}")
                nifti_save(data=np.angle(img[:, :, :, i]), img_aff=aff, path_to_dir=path_out,
                           file_name=f"naive_recon_phase_ch-{i}")
        # do rSoS
        img_rsos = np.zeros_like(np.abs(img[:, :, :, 0]))
        for i in tqdm.trange(img.shape[2], desc="rsos"):
            img_rsos[:, :, i] = root_sum_of_squares(img[:, :, i], dim_channel=-2)
        # nifti save
        nifti_save(data=img_rsos, img_aff=aff, path_to_dir=path_out, file_name="naive_rsos_recon")
        nifti_save(data=k_sampling_mask.astype(float), img_aff=aff, path_to_dir=path_out, file_name="sampling_pattern")

    # save as torch tensor for recon
    if echoes_bu is not None:
        torch_save(k_space[..., echoes_bu], path_to_file=path_out, file_name="k_space_bu")
        torch_save(k_sampling_mask[..., echoes_bu], path_to_file=path_out, file_name="k_sampling_mask_bu")
        torch_save(echoes_bu, path_to_file=path_out, file_name="echo_nums_bu")
    if echoes_bd is not None:
        torch_save(k_space[..., echoes_bd], path_to_file=path_out, file_name="k_space_bd")
        torch_save(k_sampling_mask[..., echoes_bd], path_to_file=path_out, file_name="k_sampling_mask_bd")
        torch_save(echoes_bd, path_to_file=path_out, file_name="echo_nums_bd")
    if echoes_bd is None and echoes_bu is None:
        torch_save(k_space, path_out, "k_space")
        torch_save(k_sampling_mask, path_out, "k_sampling_mask")
    torch_save(aff, path_out, "affine")
    torch_save(noise_scans, path_out, "k_noise_scans")


def pulseq():
    # setup logging
    setup_program_logging(name="Raw Data to torch", level=logging.INFO)

    # setup parser
    parser, args = setup_parser(prog_name="Raw Data to torch", dict_config_dataclasses={"settings": RD})

    # get config
    rd_config = RD.from_cli(args=args.settings, parser=parser)
    rd_config.display()

    try:
        pulseq_rd_to_torch(config=rd_config)
    except Exception as e:
        parser.print_help()
        logging.exception(e)


def siemens_rd_to_torch(config: RD):
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
        # twix_hl = twixtools.map_twix(path_to_file.as_posix())
    # noise = twix_hl[0]["image"]
    # data = twix_hl[1]["image"]
    # refscan = twix_hl[1]["refscan"]
    # data.flags["remove_os"] = True
    # refscan.flags["remove_os"] = True
    #
    # k_space = data[:].squeeze()
    # k_space = np.permute_dims(k_space, [1, 2, 4, 3, 0])
    # k_ref = refscan[:].squeeze()
    # k_ref = np.permute_dims(k_ref, [1, 2, 4, 3, 0])
    #
    # noise_scans = noise[:].squeeze()

    geometry = twix["geometry"]
    data_mdbs = twix["mdb"]
    hdr = twix["hdr"]
    log_module.info("Loading RD")
    k_space, k_sampling_mask, aff, noise_scans = load_siemens_rd(
        data_mdbs=data_mdbs, hdr=hdr,
        geometry=geometry,
        device=device
    )

    log_module.info("Saving")
    # output path
    path_out = plib.Path(config.out_path).absolute()

    if config.visualize:
        # transform into image
        img = ifft(k_space, dims=(0, 1))
        # do rSoS
        img = np.squeeze(root_sum_of_squares(img, dim_channel=-2))
        # nifti save
        nifti_save(data=img, img_aff=aff, path_to_dir=path_out, file_name="naive_rsos_recon")
        nifti_save(data=k_sampling_mask.astype(float)[:, :, :, 0], img_aff=aff, path_to_dir=path_out,
                   file_name="sampling_pattern")

    # save as torch tensor for recon
    torch_save(k_space, path_out, "k_space")
    torch_save(aff, path_out, "affine")
    torch_save(noise_scans, path_out, "k_noise_scans")
    torch_save(k_sampling_mask, path_out, "k_sampling_mask")


def siemens():
    # setup logging
    setup_program_logging(name="Raw Data to torch", level=logging.INFO)

    # setup parser
    parser, args = setup_parser(prog_name="Raw Data to torch", dict_config_dataclasses={"settings": RD})

    # get config
    rd_config = RD.from_cli(args=args.settings, parser=parser)
    rd_config.display()

    try:
        siemens_rd_to_torch(config=rd_config)
    except Exception as e:
        parser.print_help()
        logging.exception(e)


if __name__ == '__main__':
    pulseq()
