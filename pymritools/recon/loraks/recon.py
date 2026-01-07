"""
Main script for reconstruction and Commandline interface

"""
import logging
import pathlib as plib

import torch

from pymritools.config import setup_program_logging, setup_parser
from pymritools.config.recon.loraks import Settings
from pymritools.recon.loraks.loraks import LoraksOptions, Loraks, LoraksImplementation, OperatorType
from pymritools.recon.loraks.ac_loraks import AcLoraksOptions, SolverType, NullspaceAlgorithm
from pymritools.recon.loraks.utils import (
    check_channel_batch_size_and_batch_channels, prepare_k_space_to_batches, unprepare_batches_to_k_space,
    pad_input, unpad_output
)
from pymritools.utils import torch_load, fft_to_img, nifti_save, torch_save, root_sum_of_squares, adaptive_combine

logger = logging.getLogger(__name__)


def get_options_from_cmd_config(config: Settings) -> LoraksOptions:
    if config.loraks_algorithm.startswith("AC-LORAKS"):
        opts = AcLoraksOptions()
        opts.loraks_type = LoraksImplementation.AC_LORAKS
        if config.loraks_algorithm.endswith("ls"):
            opts.solver_type = SolverType.LEASTSQUARES
        else:
            opts.solver_type = SolverType.AUTOGRAD
        match config.matrix_decomposition_type:
            case "lr_svd":
                opts.nullspace_algorithm = NullspaceAlgorithm.TORCH_LR
            case "sor_svd":
                opts.nullspace_algorithm = NullspaceAlgorithm.SOR_SVD
            case "rand_ns":
                opts.nullspace_algorithm = NullspaceAlgorithm.RANDOM_NS
            case "r_svd":
                opts.nullspace_algorithm = NullspaceAlgorithm.RSVD
            case _:
                opts.nullspace_algorithm = NullspaceAlgorithm.EIGH
    elif config.loraks_algorithm == "P-LORAKS":
        opts = LoraksOptions()
        opts.loraks_type = LoraksImplementation.P_LORAKS
    else:
        raise ValueError(f"Unknown loraks algorithm: {config.loraks_algorithm}")

    opts.loraks_neighborhood_size = config.patch_size
    opts.regularization_lambda = config.reg_lambda
    opts.batch_size_channels = config.batch_size
    opts.max_num_iter = config.max_num_iter
    opts.rank.value = config.rank

    # set device
    if config.use_gpu:
        device = torch.device(f"cuda:{config.gpu_device}") if torch.cuda.is_available() else "cpu"
    else:
        device = torch.device("cpu")
    opts.device = device

    if config.matrix_type == "S":
        opts.loraks_matrix_type = OperatorType.S
    elif config.matrix_type == "C":
        opts.loraks_matrix_type = OperatorType.C
    else:
        raise ValueError(f"Unknown matrix type: {config.matrix_type}")

    for key, val in opts.__dict__.items():
        logger.info(f"\t\t  {key}: {val}")
    return opts


def main(config: Settings):

    logger.info("Load Data")
    k = torch_load(config.in_k_space)
    while k.ndim < 5:
        logger.info(f"Expanding dimension of k-space to {k.ndim + 1}")
        k.unsqueeze_(-1)
    logger.info(f"Found input k-space with shape: {k.shape}")

    if plib.Path(config.in_affine).is_file():
        aff = torch_load(config.in_affine)
    else:
        aff = torch.eye(4)

    if config.process_slice:
        k = k[:, :, k.shape[2] // 2, None]

    path_out = plib.Path(config.out_path).absolute()
    if not path_out.exists():
        logger.info(f"mkdir {path_out.as_posix()}")
        path_out.mkdir(exist_ok=True, parents=True)

    img_in = fft_to_img(k[:, :, k.shape[2] // 2], dims=(0, 1))

    nifti_save(data=img_in.abs(), img_aff=aff, path_to_dir=path_out, file_name="input_img")

    logger.info("Prepare Data")
    batch_size_channels = config.batch_size
    # batching
    batch_channel_indices = check_channel_batch_size_and_batch_channels(
        k_space_rpsct=k, batch_size_channels=batch_size_channels
    )
    k_batched, input_shape = prepare_k_space_to_batches(
        k_space_rpsct=k, batch_channel_indices=batch_channel_indices
    )
    # padding
    k_batched, padding = pad_input(k_batched)

    logger.info(f"Setup")
    opts = get_options_from_cmd_config(config)
    loraks = Loraks.create(opts)

    logger.info("Reconstruction")
    k_recon = loraks.reconstruct(k_batched)
    torch.cuda.empty_cache()

    logger.info("Unprepare")
    k_recon = unpad_output(k_space=k_recon, padding=padding)
    torch.cuda.empty_cache()

    logger.info("Unbatch / Reshape")
    k_recon = unprepare_batches_to_k_space(
        k_batched=k_recon, batch_channel_indices=batch_channel_indices, original_shape=input_shape
    )

    logger.info("RSOS")
    img = fft_to_img(k_recon, dims=(0, 1))
    rsos = root_sum_of_squares(img, dim_channel=-2)

    nifti_save(
        data=rsos,
        img_aff=aff, path_to_dir=path_out, file_name="recon_img_rsos"
    )
    # nifti_save(
    #     data=img.abs().squeeze(),
    #     img_aff=aff, path_to_dir=path_out, file_name="recon_img"
    # )

    logger.info("Save")
    torch_save(data=k_recon, path_to_file=path_out, file_name="k_recon")
    torch_save(data=aff, path_to_file=path_out, file_name="affine")

    logger.info("Adaptive combine")
    ac = adaptive_combine(channel_img_data_rpsct=img, batch_size=1, use_gpu=True)

    for i, f in enumerate([torch.abs, torch.angle]):
        nifti_save(
            data=f(ac),
            img_aff=aff, path_to_dir=path_out, file_name=f"recon_img_adac_{['mag', 'phase'][i]}"
        )


def loraks_from_cli():
    setup_program_logging(name="LORAKS Reconstruction", level=logging.INFO)
    parser, prog_args = setup_parser(
        prog_name="LORAKS Reconstruction",
        dict_config_dataclasses={
            "config": Settings
        }
    )
    settings = Settings.from_cli(args=prog_args.config, parser=parser)
    settings.display()
    try:
        main(settings)
    except Exception as e:
        parser.print_usage()
        logger.exception(e)


if __name__ == '__main__':
    loraks_from_cli()
