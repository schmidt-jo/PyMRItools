import logging
import pathlib as plib

import torch

from pymritools.config.recon import PyLoraksConfig
from pymritools.config import setup_program_logging, setup_parser
from pymritools.utils import torch_save, torch_load, root_sum_of_squares, nifti_save, fft
from pymritools.processing.coil_compression import compress_channels_2d
from pymritools.recon.loraks.algorithms import ac_loraks, loraks

log_module = logging.getLogger(__name__)


def load_data(settings: PyLoraksConfig):
    logging.debug("Load data")
    k_space = torch_load(settings.in_k_space)
    affine = torch_load(settings.in_affine)
    if settings.in_sampling_mask:
        sampling_pattern = torch_load(settings.in_sampling_mask)
    else:
        sampling_pattern = (torch.abs(k_space) > 1e-9)[:, :, 0, 0]

    logging.debug(f"For debug reduce dims")
    if settings.debug:
        # for debugging take one coil
        k_space = k_space[:, :, :, 0, None, :]
        # also take one slice. if not set anyway, we set it
        settings.process_slice = True

    logging.debug(f"Check single slice toggle set")
    if settings.process_slice:
        mid_slice = int(k_space.shape[2] / 2)
        logging.info(f"single slice processing: pick slice {mid_slice + 1}")
        k_space = k_space[:, :, mid_slice, None]

    # logging.debug(f"Check sampling pattern shape")
    # # if sampling_pattern.shape.__len__() < 3:
    # #     # sampling pattern supposed to be x, y, t
    # #     sampling_pattern = sampling_pattern[:, :, None]
    device = torch.device(f"cuda:{settings.gpu_device}") if settings.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    if settings.coil_compression is not None:
        k_space = compress_channels_2d(
            input_k_space=k_space,
            sampling_pattern=sampling_pattern,
            num_compressed_channels=settings.coil_compression,
            use_ac_data=True, device=device
        )
    # get shape
    while k_space.shape.__len__() < 5:
        # probably when processing single slice or debugging
        k_space = k_space[..., None]

    return k_space, sampling_pattern, affine


def recon(settings: PyLoraksConfig, mode: str):
    # setup
    log_module.info(f"Set output path: {settings.out_path}")
    path_out = plib.Path(settings.out_path).absolute()

    path_figs = path_out.joinpath("figs/")
    if settings.visualize:
        log_module.info(f"Set figure path for visualizations: {path_figs}")
        path_figs.mkdir(parents=True, exist_ok=True)

    # set up device
    if settings.use_gpu and torch.cuda.is_available():
        logging.info(f"configuring gpu::  cuda:{settings.gpu_device}")
        device = torch.device(f"cuda:{settings.gpu_device}")
    else:
        device = torch.device("cpu")
    torch.manual_seed(0)

    # load data
    k_space, sampling_mask, affine = load_data(settings=settings)

    log_module.info(f"___ Loraks Reconstruction ___")
    log_module.info(f"Radius - {settings.radius}; ")
    log_module.info(
        f"Rank C - {settings.c_rank}; Lambda C - {settings.c_lambda}; "
        f"Rank S - {settings.s_rank}; Lambda S - {settings.s_lambda}; "
        f"coil compression - {settings.coil_compression}")

    # set up name
    loraks_name = f"{mode}_k_space_recon_r-{settings.radius}"
    if settings.c_lambda > 1e-6 and mode != "loraks":
        loraks_name = f"{loraks_name}_lc-{settings.c_lambda:.3f}_rank-c-{settings.c_rank}"
    if settings.s_lambda > 1e-6:
        loraks_name = f"{loraks_name}_ls-{settings.s_lambda:.3f}_rank-s-{settings.s_rank}"
    loraks_name = loraks_name.replace(".", "p")

    if mode == "ac-loraks":
        loraks_recon = ac_loraks(
            k_space_x_y_z_ch_t=k_space, sampling_mask_x_y_t=sampling_mask,
            radius=settings.radius,
            rank_c=settings.c_rank, lambda_c=settings.c_lambda,
            rank_s=settings.s_rank, lambda_s=settings.s_lambda,
            max_num_iter=settings.max_num_iter, conv_tol=settings.conv_tol,
            batch_size_channels=settings.batch_size,
            device=device, path_visuals=path_figs
        )
    elif mode == "loraks":
        loraks_recon = loraks(
            k_space_x_y_z_ch_t=k_space, sampling_mask_x_y_t=sampling_mask,
            radius=settings.radius,
            rank=settings.s_rank, lam=settings.s_lambda,
            max_num_iter=settings.max_num_iter, conv_tol=settings.conv_tol,
            batch_size_echoes=settings.batch_size,
            device=device
        )
    else:
        err = f"Mode {mode} not recognized or implemented."
        log_module.error(err)
        raise ValueError(err)

    if settings.process_slice:
        loraks_recon = torch.squeeze(loraks_recon)[:, :, None, :]

    logging.info(f"Save k-space reconstruction")
    # loraks_phase = torch.angle(loraks_recon)
    # loraks_phase = torch.mean(loraks_phase, dim=-2)
    # loraks_mag = torch.abs(loraks_recon)

    logging.info("FFT into image space")
    # fft into real space
    loraks_recon_img = fft(loraks_recon, img_to_k=False, axes=(0, 1))

    logging.info("rSoS channels")
    dim_channel = -2

    # for nii we rSoS combine channels
    loraks_recon_mag = root_sum_of_squares(input_data=loraks_recon_img, dim_channel=dim_channel)

    # ToDo Phase combination implementation
    # loraks_phase = torch.angle(loraks_recon_img)
    # loraks_phase = torch.mean(loraks_phase, dim=-2)

    nii_name = loraks_name.replace("k_space", "image")
    nifti_save(data=loraks_recon_mag, img_aff=affine, path_to_dir=path_out, file_name=f"{nii_name}_mag")
    # nifti_save(data=loraks_phase, img_aff=affine, path_to_dir=path_out, file_name=f"{nii_name}_phase")

    # save data as tensors, for further usage of whole data
    if not settings.process_slice:
        torch_save(data=loraks_recon, path_to_file=path_out, file_name=f"{loraks_name}_k-space")


def recon_ac_loraks():
    # setup  logging
    setup_program_logging(name="AC Loraks", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(prog_name="AC Loraks", dict_config_dataclasses={"settings": PyLoraksConfig})
    # get cli args
    settings = PyLoraksConfig.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        recon(settings=settings, mode="ac-loraks")
    except Exception as e:
        logging.exception(e)
        parser.print_help()


def recon_loraks():
    # setup  logging
    setup_program_logging(name="Loraks", level=logging.INFO)
    # setup parser
    parser, args = setup_parser(prog_name="Loraks", dict_config_dataclasses={"settings": PyLoraksConfig})
    # get cli args
    settings = PyLoraksConfig.from_cli(args=args.settings, parser=parser)
    settings.display()

    try:
        recon(settings=settings, mode="loraks")
    except Exception as e:
        logging.exception(e)
        parser.print_help()


if __name__ == '__main__':
    recon_ac_loraks()
